import os.path
from collections import OrderedDict
import re
from .ManyDepth import *
from .SconeOcc import *
from .SconeVis import *
from ..utility.utils import init_weights
from ..utility.scone_utils import initialize_scone_occ_weights, initialize_scone_vis_weights

dir_path = os.path.abspath(os.path.dirname(__file__))
macarons_weights_path = os.path.join(dir_path, "../../weights/macarons")
scone_weights_path = os.path.join(dir_path, "../../weights/scone")

# Path to the depth model with a pretrained image feature extractor (ResNet18)
pretrained_depth_model_path = os.path.join(dir_path, "../../weights/resnet/depth_with_resnet_imagenet_weights.pth")


class MacaronsWrapper:
    def __init__(self, macarons_depth, macarons_scone):
        self.depth = macarons_depth
        self.scone = macarons_scone

        self.image_height = macarons_depth.image_height
        self.image_width = macarons_depth.image_width

    def train(self):
        self.depth.train()
        self.scone.train()

    def eval(self):
        self.depth.eval()
        self.scone.eval()

    def to(self, device):
        self.depth = self.depth.to(device)
        self.scone = self.scone.to(device)

    def state_dict(self):
        state_dict = {}
        state_dict['depth'] = self.depth.state_dict()
        state_dict['scone'] = self.scone.state_dict()
        return state_dict

    def load_state_dict(self, state_dict, ddp=False, depth_only=False):
        if not ddp:
            self.depth.load_state_dict(state_dict['depth'])
            if not depth_only:
                self.scone.load_state_dict(state_dict['scone'])
        else:
            self.depth = load_ddp_state_dict(self.depth, state_dict['depth'])
            if not depth_only:
                self.scone = load_ddp_state_dict(self.scone, state_dict['scone'])

    def apply(self, init_weights):
        self.depth.apply(init_weights)
        self.scone.apply(init_weights)


class MacaronsOptimizer:
    def __init__(self, depth_optimizer, scone_optimizer, freeze_scone=True, freeze_depth=False):
        self.depth = depth_optimizer
        self.scone = scone_optimizer

        self.freeze_scone = freeze_scone
        self.freeze_depth = freeze_depth

    def zero_grad(self):
        if not self.freeze_depth:
            self.depth.zero_grad()
        if not self.freeze_scone:
            self.scone.zero_grad()

    def step(self):
        if not self.freeze_depth:
            self.depth.step()
        if not self.freeze_scone:
            self.scone.step()

    def state_dict(self):
        state_dict = {}
        state_dict['depth'] = self.depth.state_dict()
        state_dict['scone'] = self.scone.state_dict()
        return state_dict

    def load_state_dict(self, state_dict, depth_only=False):
        self.depth.load_state_dict(state_dict['depth'])
        if not depth_only:
            self.scone.load_state_dict(state_dict['scone'])


class Macarons(nn.Module):
    def __init__(self, depth_model, occupancy_model, visibility_model):
        """
        Wrapper class for MACARONS model.

        :param depth_model: (ManyDepth) Depth prediction model inspired by ManyDepth.
        :param occupancy_model: (SconeOcc) Occupancy probability prediction model from SCONE.
        :param visibility_model: (SconeVis) Visibility gain prediction model from SCONE.
        """
        super(Macarons, self).__init__()

        self.depth = depth_model
        self.occupancy = occupancy_model
        self.visibility = visibility_model

        if depth_model is not None:
            self.image_height = depth_model.input_height
            self.image_width = depth_model.input_width

    def forward(self, mode,
                x=None, x_alpha=None, R=None, T=None, zfar=None, device=None, gt_pose=None,  # Depth args
                partial_point_cloud=None, proxy_points=None, view_harmonics=None  # SCONE args
                ):

        if mode == 'depth':
            if (x is None) or (x_alpha is None) or (R is None) or (T is None) or (zfar is None) or (device is None):
                raise NameError("For 'occupancy' mode, you should provide the following args:"
                                "x, x_alpha, R, T, zfar, device")
            res = self.depth(x=x, x_alpha=x_alpha, R=R, T=T, zfar=zfar, device=device, gt_pose=gt_pose)

        elif mode == 'occupancy':
            if (partial_point_cloud is None) or (proxy_points is None) or (view_harmonics is None):
                raise NameError("For 'occupancy' mode, you should provide the following args:"
                                "partial_point_cloud, proxy_points, view_harmonics")
            res = self.occupancy(pc=partial_point_cloud, x=proxy_points, view_harmonics=view_harmonics)

        elif mode == 'visibility':
            if (proxy_points is None) or (view_harmonics is None):
                raise NameError("For 'visibility' mode, you should provide the following args:"
                                "proxy_points, view_harmonics")
            res = self.visibility(proxy_points, view_harmonics=view_harmonics)

        else:
            raise NameError("Invalid mode. Please select a mode between 'depth', 'occupancy' and 'visibility'.")

        return res

    def compute_visibility_gains(self, pts, harmonics, X_cam):
        """
        Compute visibility gains of each points in pts for each camera in X_cam.
        :param pts: (Tensor) Input point cloud. Tensor with shape (n_clouds, seq_len, pts_dim)
        :param harmonics: (Tensor) Predicted visibility gain functions as coordinates in spherical harmonics.
        Has shape (n_clouds, seq_len, n_harmonics).
        :param X_cam: (Tensor) Tensor of camera centers' positions, with shape (n_clouds, n_camera_candidates, 3)
        :return: (Tensor) The predicted per-point visibility gains of all points.
        Has shape (n_clouds, n_camera_candidates, seq_len)
        """
        clear_spherical_harmonics_cache()
        n_clouds = pts.shape[0]
        seq_len = pts.shape[1]
        n_harmonics = self.visibility.n_harmonics
        n_camera_candidates = X_cam.shape[1]

        device = pts.get_device()
        if device < 0:
            device = "cpu"

        X_pts = pts[..., :3]

        rays = (X_cam.view(n_clouds, n_camera_candidates, 1, 3).expand(-1, -1, seq_len, -1)
                - X_pts.view(n_clouds, 1, seq_len, 3).expand(-1, n_camera_candidates, -1, -1)).view(-1, 3)
        _, theta, phi = get_spherical_coords(rays)
        theta = -theta + np.pi / 2.

        z = torch.zeros([i for i in theta.shape] + [0], device=device)
        for i in range(self.visibility.max_harmonic_rank):
            y = get_spherical_harmonics(l=i, theta=theta, phi=phi)
            z = torch.cat((z, y), dim=-1)
        z = z.view(n_clouds, n_camera_candidates, seq_len, n_harmonics)

        z = torch.sum(z * harmonics.view(n_clouds, 1, seq_len, 64).expand(-1, n_camera_candidates, -1, -1), dim=-1)
        if self.visibility.use_sigmoid:
            z = torch.sigmoid(z)
        else:
            z = torch.relu(z)
            raise NameError("WARNING! ReLU has been used in visibility model.")

        return z


def load_ddp_state_dict(model, state_dict):
    """
    Function to load a DDP state dictionary into a non-DDP model.
    :param model
    :param state_dict
    :return: None
    """
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k, v in state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v

        else:
            model_dict = state_dict

    model.load_state_dict(model_dict)

    return model


def create_macarons_model_old(device, learn_pose=False):
    print("Creating depth model...")
    # depth_model = create_many_depth_model(learn_pose=learn_pose, device=device)
    depth_model = torch.load(pretrained_depth_model_path, map_location=device)

    print("\nCreating occupancy model...")
    occupancy_model = SconeOcc().to(device)

    print("\nCreating visibility model...")
    visibility_model = SconeVis().to(device)

    model = Macarons(depth_model=depth_model,
                     occupancy_model=occupancy_model,
                     visibility_model=visibility_model).to(device)

    return model


def create_macarons_depth(device):
    print("Creating depth model...")
    # depth_model = create_many_depth_model(learn_pose=learn_pose, device=device)
    depth_model = torch.load(pretrained_depth_model_path, map_location=device)

    macarons_depth = Macarons(depth_model=depth_model,
                              occupancy_model=None,
                              visibility_model=None).to(device)

    return macarons_depth


def create_macarons_model(device, learn_pose=False):
    print("Creating depth model...")
    # depth_model = create_many_depth_model(learn_pose=learn_pose, device=device)
    depth_model = torch.load(pretrained_depth_model_path, map_location=device, weights_only=False)

    print("\nCreating occupancy model...")
    occupancy_model = SconeOcc().to(device)

    print("\nCreating visibility model...")
    visibility_model = SconeVis().to(device)

    macarons_depth = Macarons(depth_model=depth_model,
                              occupancy_model=None,
                              visibility_model=None).to(device)

    macarons_scone = Macarons(depth_model=None,
                              occupancy_model=occupancy_model,
                              visibility_model=visibility_model).to(device)

    model = MacaronsWrapper(macarons_depth=macarons_depth, macarons_scone=macarons_scone)

    return model


def load_pretrained_module_for_macarons(pretrained_model_path, device):
    model = torch.load(pretrained_model_path, map_location=device).to(device)
    print("Model path:", pretrained_model_path)
    return model.float()


def load_pretrained_module_weights_for_macarons(module, pretrained_weights_path, ddp_model, device):
    """
    Loads an already trained model for inference on a single GPU.

    :param module: Vanilla module.
    :param trained_model_name: (str) Name of trained model's checkpoint.
    :param ddp_model: (bool) Set to True to load a model trained with DDP.
    :param device: Device.
    :return:
    """
    module = module.to(device)

    # Loads checkpoint
    checkpoint = torch.load(pretrained_weights_path, map_location=device)
    print("Model name:", pretrained_weights_path)
    print("Trained for", checkpoint['epoch'], 'epochs.')
    print("Training finished with loss", checkpoint['loss'])

    # Loads trained weights
    if ddp_model:
        module = load_ddp_state_dict(module, checkpoint['model_state_dict'])
    else:
        module.load_state_dict(checkpoint['model_state_dict'])

    return module


def load_pretrained_depth_weights_from_macarons(model, pretrained_weights_path, ddp_model, device):
    """
    Loads an already trained model for inference on a single GPU.

    :param module: Vanilla module.
    :param trained_model_name: (str) Name of trained model's checkpoint.
    :param ddp_model: (bool) Set to True to load a model trained with DDP.
    :param device: Device.
    :return:
    """
    model.to(device)

    # Loads checkpoint
    checkpoint = torch.load(pretrained_weights_path, map_location=device)
    print("Model name:", pretrained_weights_path)
    print("Trained for", checkpoint['epoch'], 'epochs.')
    print("Training finished with loss", checkpoint['loss'])

    model.load_state_dict(checkpoint['model_state_dict'], ddp=ddp_model, depth_only=True)

    return model
