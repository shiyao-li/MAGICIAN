from typing import List
import numpy as np
import torch
import math
from pytorch3d.renderer import FoVPerspectiveCameras as P3DCameras
from pytorch3d.renderer.cameras import _get_sfm_calibration_matrix
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from pytorch3d.ops import knn_points

def getProjectionMatrix(znear, zfar, fovX, fovY):
    if isinstance(fovX, torch.Tensor):
        tanHalfFovX = torch.tan((fovX / 2))
    else:
        tanHalfFovX = math.tan((fovX / 2))
    
    if isinstance(fovY, torch.Tensor):
        tanHalfFovY = torch.tan((fovY / 2))
    else:
        tanHalfFovY = math.tan((fovY / 2))
    

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    if isinstance(fovX, torch.Tensor):
        P = torch.zeros(4, 4, device=fovX.device)
    else:
        P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P



def getWorld2View2(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    if type(R) == np.ndarray:
        _translate = translate.cpu().numpy()
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0

        C2W = np.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + _translate) * scale
        C2W[:3, 3] = cam_center
        Rt = np.linalg.inv(C2W)
        return np.float32(Rt)
    else:
        translate = translate.to(R.device)
        Rt = torch.zeros((4, 4), device=R.device)
        Rt[:3, :3] = R.transpose(-1, -2)
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0

        C2W = torch.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        Rt = torch.linalg.inv(C2W)
    return Rt

def fov2focal(fov, pixels):
    if isinstance(fov, torch.Tensor) or isinstance(pixels, torch.Tensor):
        return pixels / (2 * torch.tan(fov / 2))
    else:
        return pixels / (2 * math.tan(fov / 2))

def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm

def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )
    
def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V

def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def focal2fov(focal, pixels):
    if isinstance(focal, torch.Tensor) or isinstance(pixels, torch.Tensor):
        return 2 * torch.atan(pixels / (2 * focal))
    else:
        return 2 * math.atan(pixels / (2 * focal))

debug = True

def rescale_cameras(
    cameras, 
    scale_factor:float,
    add_extension_to_image_name:str='',
    no_original_image:bool=False,
):
    gs_cameras = cameras.gs_cameras
    new_gs_cameras = [] 
    for gs_camera in gs_cameras:    
        new_gs_cameras.append(
            GSCamera(
                colmap_id=gs_camera.colmap_id, 
                image=None if no_original_image else gs_camera.original_image, 
                gt_alpha_mask=None,
                R=gs_camera.R, 
                T=gs_camera.T * scale_factor,
                FoVx=gs_camera.FoVx, 
                FoVy=gs_camera.FoVy,
                image_name=gs_camera.image_name + add_extension_to_image_name,
                uid=gs_camera.uid,
                image_height=gs_camera.image_height if no_original_image else None, 
                image_width=gs_camera.image_width if no_original_image else None,
            )
        )
    return CamerasWrapper(new_gs_cameras)


def interpolate_cameras(
    p3d_cameras_1:P3DCameras, 
    p3d_cameras_2:P3DCameras, 
    t:float=0.5):
    new_p3d_cameras = p3d_cameras_1.clone()
    quaternions_1 = matrix_to_quaternion(p3d_cameras_1.R)
    quaternions_2 = matrix_to_quaternion(p3d_cameras_2.R)
    
    new_p3d_cameras.K = (1. - t) * p3d_cameras_1.K + t * p3d_cameras_2.K
    new_p3d_cameras.T = (1. - t) * p3d_cameras_1.T + t * p3d_cameras_2.T
    new_quaternions = (1. - t) * quaternions_1 + t * quaternions_2
    new_p3d_cameras.R = quaternion_to_matrix(torch.nn.functional.normalize(new_quaternions, dim=-1))
    
    return new_p3d_cameras


class GSCamera(torch.nn.Module):
    """Class to store Gaussian Splatting camera parameters.
    """
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 image_height=None, image_width=None,
                 detach=True,
                 ):
        """
        Args:
            colmap_id (int): ID of the camera in the COLMAP reconstruction.
            R (np.array): Rotation matrix.
            T (np.array): Translation vector.
            FoVx (float): Field of view in the x direction.
            FoVy (float): Field of view in the y direction.
            image (np.array): GT image.
            gt_alpha_mask (_type_): _description_
            image_name (_type_): _description_
            uid (_type_): _description_
            trans (_type_, optional): _description_. Defaults to np.array([0.0, 0.0, 0.0]).
            scale (float, optional): _description_. Defaults to 1.0.
            data_device (str, optional): _description_. Defaults to "cuda".
            image_height (_type_, optional): _description_. Defaults to None.
            image_width (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        super(GSCamera, self).__init__()

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.uid = uid
        self.colmap_id = colmap_id
        
        if True:
            if type(R) == np.ndarray:
                self.R = torch.tensor(R).to(self.data_device)
                self.T = torch.tensor(T).to(self.data_device)
            else:
                if detach:
                    self.R = R.clone().detach().to(self.data_device)
                    self.T = T.clone().detach().to(self.data_device)
                    self.R0 = self.R.clone()
                    self.T0 = self.T.clone()
                else:
                    self.R = R.clone().to(self.data_device)
                    self.T = T.clone().to(self.data_device)
                    self.R0 = self.R.clone().detach()
                    self.T0 = self.T.clone().detach()
        else:
            self.R = R.clone()
            self.T = T.clone()
            self.R0 = R.clone()
            self.T0 = T.clone()
        
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        
        if image is None:
            if image_height is None or image_width is None:
                raise ValueError("Either image or image_height and image_width must be specified")
            else:
                self.image_height = image_height
                self.image_width = image_width
        else:        
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]

            if gt_alpha_mask is not None:
                self.original_image *= gt_alpha_mask.to(self.data_device)
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        self.gt_alpha_mask = gt_alpha_mask
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = torch.tensor(trans).to(self.data_device)
        self.scale = scale
        
        self.prcppoint = torch.tensor([0.5, 0.5], dtype=torch.float32).to(self.data_device)
        
        tan_fovx = np.tan(self.FoVx / 2.)
        tan_fovy = np.tan(self.FoVy / 2.)
        self.focal_x = self.image_width / (2. * tan_fovx)
        self.focal_y = self.image_height / (2. * tan_fovy)
        
    @property
    def device(self):
        return self.data_device
    
    @property   
    def world_view_transform(self):
        # return getWorld2View2(self.R, self.T, self.trans, self.scale).transpose(0, 1)
        return getWorld2View2(self.R, self.T).transpose(0, 1)
    
    @property
    def projection_matrix(self):
        return getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
    
    @property
    def full_proj_transform(self):
        return (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
    
    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]
    
    def get_camera_center(self):
        return self.camera_center
    
    def to(self, device):
        self.world_view_transform = self.world_view_transform.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
        self.full_proj_transform = self.full_proj_transform.to(device)
        self.camera_center = self.camera_center.to(device)
        return self
    
    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)
    
    def update_pose(self, cam_r_delta, cam_t_delta):
        tau = torch.cat([cam_t_delta, cam_r_delta], axis=0)

        T_w2c = torch.eye(4, device=tau.device)
        T_w2c[0:3, 0:3] = self.R
        T_w2c[0:3, 3] = self.T

        new_w2c = SE3_exp(tau) @ T_w2c

        new_R = new_w2c[0:3, 0:3]
        new_T = new_w2c[0:3, 3]

        self.update_RT(new_R, new_T)
        
    def transform_points_world_to_view(
        self, points:torch.Tensor, use_p3d_convention:bool=True,
    ):
        """_summary_

        Args:
            points (torch.Tensor): Shape (N, 3).
            use_p3d_convention (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (N, 4)
        view_points = (points_h @ self.world_view_transform)[..., :3]  # (N, 3)
        if use_p3d_convention:
            factors = torch.tensor([[-1, -1, 1]], device=points.device)  # (1, 3)
            view_points = factors * view_points  # (N, 3)
        return view_points
    
    
    def project_points(self, points:torch.Tensor, use_p3d_convention:bool=True):
        """_summary_

        Args:
            points (torch.Tensor): Shape (N, 3).
            use_p3d_convention (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (N, 4)
        proj_points = points_h @ self.full_proj_transform  # (N, 4)
        proj_points = proj_points[..., :2] / proj_points[..., 3:4]  # (N, 2)
        if use_p3d_convention:
            height, width = self.image_height, self.image_width
            factors = torch.tensor([[-width / min(height, width), -height / min(height, width)]], device=points.device)  # (1, 2)
            proj_points = factors * proj_points  # (N, 2)
        return proj_points

    
def create_p3d_cameras(R=None, T=None, K=None, znear=0.0001):
    """Creates pytorch3d-compatible camera object from R, T, K matrices.

    Args:
        R (torch.Tensor, optional): Rotation matrix. Defaults to Identity.
        T (torch.Tensor, optional): Translation vector. Defaults to Zero.
        K (torch.Tensor, optional): Camera intrinsics. Defaults to None.
        znear (float, optional): Near clipping plane. Defaults to 0.0001.

    Returns:
        pytorch3d.renderer.cameras.FoVPerspectiveCameras: pytorch3d-compatible camera object.
    """
    if R is None:
        R = torch.eye(3)[None]
    if T is None:
        T = torch.zeros(3)[None]
        
    if K is not None:
        p3d_cameras = P3DCameras(R=R, T=T, K=K, znear=0.0001)
    else:
        p3d_cameras = P3DCameras(R=R, T=T, znear=0.0001)
        p3d_cameras.K = p3d_cameras.get_projection_transform().get_matrix().transpose(-1, -2)
        
    return p3d_cameras


def convert_camera_from_gs_to_pytorch3d(gs_cameras, device='cuda'):
    """
    From Gaussian Splatting camera parameters,
    computes R, T, K matrices and outputs pytorch3d-compatible camera object.

    Args:
        gs_cameras (List of GSCamera): List of Gaussian Splatting cameras.
        device (_type_, optional): _description_. Defaults to 'cuda'.

    Returns:
        p3d_cameras: pytorch3d-compatible camera object.
    """
    
    N = len(gs_cameras)
    
    # TODO: Directly use the torch matrix, without converting into array.
    R = torch.Tensor(np.array([gs_camera.R.cpu().numpy() for gs_camera in gs_cameras])).to(device)
    T = torch.Tensor(np.array([gs_camera.T.cpu().numpy() for gs_camera in gs_cameras])).to(device)
    fx = torch.Tensor(np.array([fov2focal(gs_camera.FoVx, gs_camera.image_width) for gs_camera in gs_cameras])).to(device)
    fy = torch.Tensor(np.array([fov2focal(gs_camera.FoVy, gs_camera.image_height) for gs_camera in gs_cameras])).to(device)
    image_height = torch.tensor(np.array([gs_camera.image_height for gs_camera in gs_cameras]), dtype=torch.int).to(device)
    image_width = torch.tensor(np.array([gs_camera.image_width for gs_camera in gs_cameras]), dtype=torch.int).to(device)
    cx = image_width / 2.  # torch.zeros_like(fx).to(device)
    cy = image_height / 2.  # torch.zeros_like(fy).to(device)
    
    w2c = torch.zeros(N, 4, 4).to(device)
    w2c[:, :3, :3] = R.transpose(-1, -2)
    w2c[:, :3, 3] = T
    w2c[:, 3, 3] = 1
    
    c2w = w2c.inverse()
    c2w[:, :3, 1:3] *= -1
    c2w = c2w[:, :3, :]
    
    distortion_params = torch.zeros(N, 6).to(device)
    camera_type = torch.ones(N, 1, dtype=torch.int32).to(device)

    # Pytorch3d-compatible camera matrices
    # Intrinsics
    image_size = torch.Tensor(
        [image_width[0], image_height[0]],
    )[
        None
    ].to(device)
    # image_size = torch.cat([image_width.view(-1, 1), image_height.view(-1, 1)], dim=-1)
    
    scale = image_size.min(dim=1, keepdim=True)[0] / 2.0
    c0 = image_size / 2.0
    
    # p0_pytorch3d = (
    #     -(
    #         torch.Tensor(
    #             (cx[0], cy[0]),
    #         )[
    #             None
    #         ].to(device)
    #         - c0
    #     )
    #     / scale
    # )
    p0_pytorch3d = -(torch.cat([cx.view(-1, 1), cy.view(-1, 1)], dim=-1) - c0) / scale
    
    # focal_pytorch3d = (
    #     torch.Tensor([fx[0], fy[0]])[None].to(device) / scale
    # )
    focal_pytorch3d = torch.cat([fx.view(-1, 1), fy.view(-1, 1)], dim=-1) / scale
    
    # print("focalp3d, p0p3d:", focal_pytorch3d.shape, p0_pytorch3d.shape)

    K = _get_sfm_calibration_matrix(
        N, "cpu", focal_pytorch3d, p0_pytorch3d, orthographic=False
    )
    # print("K:", K.shape)
    # K = K.expand(N, -1, -1)
    if K.shape[0] != N:
        raise ValueError("K shape does not match the number of cameras.")

    # Extrinsics
    line = torch.Tensor([[0.0, 0.0, 0.0, 1.0]]).to(device).expand(N, -1, -1)
    cam2world = torch.cat([c2w, line], dim=1)
    world2cam = cam2world.inverse()
    R, T = world2cam.split([3, 1], dim=-1)
    R = R[:, :3].transpose(1, 2) * torch.Tensor([-1.0, 1.0, -1]).to(device)
    T = T.squeeze(2)[:, :3] * torch.Tensor([-1.0, 1.0, -1]).to(device)

    p3d_cameras = P3DCameras(device=device, R=R, T=T, K=K, znear=0.0001)

    return p3d_cameras


def convert_camera_from_pytorch3d_to_gs(
    p3d_cameras: P3DCameras,
    height: float,
    width: float,
    device='cuda',
):
    """From a pytorch3d-compatible camera object and its camera matrices R, T, K, and width, height,
    outputs Gaussian Splatting camera parameters.

    Args:
        p3d_cameras (P3DCameras): R matrices should have shape (N, 3, 3),
            T matrices should have shape (N, 3, 1),
            K matrices should have shape (N, 3, 3).
        height (float): _description_
        width (float): _description_
        device (_type_, optional): _description_. Defaults to 'cuda'.
    """

    N = p3d_cameras.R.shape[0]
    if device is None:
        device = p3d_cameras.device

    if type(height) == torch.Tensor:
        height = int(torch.Tensor([[height.item()]]).to(device))
        width = int(torch.Tensor([[width.item()]]).to(device))
    else:
        height = int(height)
        width = int(width)

    # Inverse extrinsics
    R_inv = (p3d_cameras.R * torch.Tensor([-1.0, 1.0, -1]).to(device)).transpose(-1, -2)
    T_inv = (p3d_cameras.T * torch.Tensor([-1.0, 1.0, -1]).to(device)).unsqueeze(-1)
    world2cam_inv = torch.cat([R_inv, T_inv], dim=-1)
    line = torch.Tensor([[0.0, 0.0, 0.0, 1.0]]).to(device).expand(N, -1, -1)
    world2cam_inv = torch.cat([world2cam_inv, line], dim=-2)
    cam2world_inv = world2cam_inv.inverse()
    camera_to_worlds_inv = cam2world_inv[:, :3]

    # Inverse intrinsics
    image_size = torch.Tensor(
        [width, height],
    )[
        None
    ].to(device)
    scale = image_size.min(dim=1, keepdim=True)[0] / 2.0
    c0 = image_size / 2.0
    
    gs_cameras = []
    
    for cam_idx in range(N):
        K_inv = p3d_cameras.K[cam_idx] * scale
        fx_inv, fy_inv = K_inv[0, 0], K_inv[1, 1]
        cx_inv, cy_inv = c0[0, 0] - K_inv[0, 2], c0[0, 1] - K_inv[1, 2]
        
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = camera_to_worlds_inv[cam_idx]
        c2w = torch.cat([c2w, torch.Tensor([[0, 0, 0, 1]]).to(device)], dim=0).cpu().numpy() #.transpose(-1, -2)
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        
        image_height=height
        image_width=width
        
        fx = fx_inv.item()
        fy = fy_inv.item()
        fovx = focal2fov(fx, image_width)
        fovy = focal2fov(fy, image_height)

        FovY = fovy 
        FovX = fovx
        
        name = 'image_' + str(cam_idx)
        
        camera = GSCamera(
            colmap_id=cam_idx, image=None, gt_alpha_mask=None,
            R=R, T=T, FoVx=FovX, FoVy=FovY,
            image_name=name, uid=cam_idx,
            image_height=image_height, 
            image_width=image_width,
            )
        gs_cameras.append(camera)

    return gs_cameras


class CamerasWrapper:
    """Class to wrap Gaussian Splatting camera parameters 
    and facilitates both usage and integration with PyTorch3D.
    """
    def __init__(
        self,
        gs_cameras,
        p3d_cameras=None,
        p3d_cameras_computed=False,
        no_p3d_cameras=False,
    ) -> None:
        """
        Args:
            camera_to_worlds (_type_): _description_
            fx (_type_): _description_
            fy (_type_): _description_
            cx (_type_): _description_
            cy (_type_): _description_
            width (_type_): _description_
            height (_type_): _description_
            distortion_params (_type_): _description_
            camera_type (_type_): _description_
        """

        self.gs_cameras = gs_cameras
        
        self.no_p3d_cameras = no_p3d_cameras
        if no_p3d_cameras:
            self._p3d_cameras = None
            self._p3d_cameras_computed = True
        else:
            self._p3d_cameras = p3d_cameras
            self._p3d_cameras_computed = p3d_cameras_computed
        
        # print(gs_cameras[0])
        # print("R=", gs_cameras[0].R.device)
        # print("T=", gs_cameras[0].T.device)
        # print("wvt=", gs_cameras[0].world_view_transform)
        device = gs_cameras[0].device        
        N = len(gs_cameras)
        R = torch.Tensor(np.array([gs_camera.R.cpu().numpy() for gs_camera in gs_cameras])).to(device)
        T = torch.Tensor(np.array([gs_camera.T.cpu().numpy() for gs_camera in gs_cameras])).to(device)
        self.fx = torch.Tensor(np.array([fov2focal(gs_camera.FoVx, gs_camera.image_width) for gs_camera in gs_cameras])).to(device)
        self.fy = torch.Tensor(np.array([fov2focal(gs_camera.FoVy, gs_camera.image_height) for gs_camera in gs_cameras])).to(device)
        self.height = torch.tensor(np.array([gs_camera.image_height for gs_camera in gs_cameras]), dtype=torch.int).to(device)
        self.width = torch.tensor(np.array([gs_camera.image_width for gs_camera in gs_cameras]), dtype=torch.int).to(device)
        self.cx = self.width / 2.  # torch.zeros_like(fx).to(device)
        self.cy = self.height / 2.  # torch.zeros_like(fy).to(device)
        
        w2c = torch.zeros(N, 4, 4).to(device)
        w2c[:, :3, :3] = R.transpose(-1, -2)
        w2c[:, :3, 3] = T
        w2c[:, 3, 3] = 1
        
        c2w = w2c.inverse()
        c2w[:, :3, 1:3] *= -1
        c2w = c2w[:, :3, :]
        self.camera_to_worlds = c2w

    @classmethod
    def from_p3d_cameras(
        cls,
        p3d_cameras,
        width: float,
        height: float,
    ) -> None:
        """Initializes CamerasWrapper from pytorch3d-compatible camera object.

        Args:
            p3d_cameras (_type_): _description_
            width (float): _description_
            height (float): _description_

        Returns:
            _type_: _description_
        """
        cls._p3d_cameras = p3d_cameras
        cls._p3d_cameras_computed = True

        gs_cameras = convert_camera_from_pytorch3d_to_gs(
            p3d_cameras,
            height=height,
            width=width,
        )

        return cls(
            gs_cameras=gs_cameras,
            p3d_cameras=p3d_cameras,
            p3d_cameras_computed=True,
        )

    @property
    def device(self):
        return self.camera_to_worlds.device

    @property
    def p3d_cameras(self):
        if self.no_p3d_cameras:
            raise ValueError("No Pytorch3D cameras available.")
        
        if not self._p3d_cameras_computed:
            self._p3d_cameras = convert_camera_from_gs_to_pytorch3d(
                self.gs_cameras,
            )
            self._p3d_cameras_computed = True

        return self._p3d_cameras

    def __len__(self):
        return len(self.gs_cameras)

    def to(self, device):
        self.camera_to_worlds = self.camera_to_worlds.to(device)
        self.fx = self.fx.to(device)
        self.fy = self.fy.to(device)
        self.cx = self.cx.to(device)
        self.cy = self.cy.to(device)
        self.width = self.width.to(device)
        self.height = self.height.to(device)
        
        for gs_camera in self.gs_cameras:
            gs_camera.to(device)

        if self._p3d_cameras_computed:
            self._p3d_cameras = self._p3d_cameras.to(device)

        return self
        
    def get_spatial_extent(self):
        """Returns the spatial extent of the cameras, computed as half
        the extent of the bounding box containing all camera centers.

        Returns:
            (float): Spatial extent of the cameras.
        """
        if self.no_p3d_cameras:
            camera_centers = torch.cat([gs_camera.camera_center.view(1, 3) for gs_camera in self.gs_cameras], dim=0)
        else:
            camera_centers = self.p3d_cameras.get_camera_center()
        avg_camera_center = camera_centers.mean(dim=0, keepdim=True)
        half_diagonal = torch.norm(camera_centers - avg_camera_center, dim=-1).max().item()

        radius = 1.1 * half_diagonal
        return radius
    
    def get_neighbor_cameras(
        self,
        camera_idx:int=None,
        n_neighbors:int=5,
        position=None,
        return_idx=False,
    ):
        """Returns the n_neighbors nearest cameras (in the 3D space) to the camera at camera_idx.
        If a position is provided, it will return the n_neighbors nearest cameras to that position instead.

        Args:
            camera_idx (int): Index of the camera to find neighbors for.
            n_neighbors (int, optional): Number of neighbors to return. Defaults to 5.
            position (torch.Tensor, optional): Position to find neighbors for. Defaults to None. Has shape (3,).
            return_idx (bool, optional): Whether to return the indices of the neighbors, or the cameras themselves. Defaults to False.

        Returns:
            CamerasWrapper: A wrapper containing the neighbor cameras or their indices.
        """
        camera_centers = self.p3d_cameras.get_camera_center()
        
        if position is not None:
            current_camera_center = position.view(1, 3)
        else:
            if camera_idx is None:
                raise ValueError("Either camera_idx or position must be provided.")
            current_camera_center = camera_centers[camera_idx:camera_idx + 1]
            
        knn_camera_indices = knn_points(current_camera_center[None], camera_centers[None], K=n_neighbors).idx[0, 0]
        neighbor_gs_cameras = [self.gs_cameras[i] for i in knn_camera_indices]
        if return_idx:
            return knn_camera_indices
        return CamerasWrapper(neighbor_gs_cameras)
    
    def transform_points_world_to_view(
        self,
        points:torch.Tensor,
        use_p3d_convention:bool=True,
    ):
        """_summary_

        Args:
            points (torch.Tensor): Should have shape (n_cameras, N, 3).
            use_p3d_convention (bool, optional): Defaults to True.
        """
        world_view_transforms = torch.stack([gs_camera.world_view_transform for gs_camera in self.gs_cameras], dim=0)  # (n_cameras, 4, 4)
        
        points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (n_cameras, N, 4)
        view_points = (points_h @ world_view_transforms)[..., :3]  # (n_cameras, N, 3)
        if use_p3d_convention:
            factors = torch.tensor([[[-1, -1, 1]]], device=points.device)  # (1, 1, 3)
            view_points = factors * view_points  # (n_cameras, N, 3)
        return view_points
        
    def project_points(
        self,
        points:torch.Tensor,
        points_are_already_in_view_space:bool=False,
        use_p3d_convention:bool=True,
        znear=1e-6,
    ):
        """_summary_

        Args:
            points (torch.Tensor): Should have shape (n_cameras, N, 3).
            use_p3d_convention (bool, optional): Defaults to True.

        Returns:
            _type_: _description_
        """
        if points_are_already_in_view_space:
            full_proj_transforms = torch.stack([gs_camera.projection_matrix for gs_camera in self.gs_cameras])  # (n_depth, 4, 4)
        else:
            full_proj_transforms = torch.stack([gs_camera.full_proj_transform for gs_camera in self.gs_cameras])  # (n_cameras, 4, 4)
        
        points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (n_cameras, N, 4)
        proj_points = points_h @ full_proj_transforms  # (n_cameras, N, 4)
        proj_points = proj_points[..., :2] / proj_points[..., 3:4].clamp_min(znear)  # (n_cameras, N, 2)
        if use_p3d_convention:
            height, width = self.gs_cameras[0].image_height, self.gs_cameras[0].image_width
            # TODO: Handle different image sizes for different cameras
            factors = torch.tensor([[[-width / min(height, width), -height / min(height, width)]]], device=points.device)  # (1, 1, 2)
            proj_points = factors * proj_points  # (n_cameras, N, 2)
            if points_are_already_in_view_space:
                proj_points = - proj_points
        return proj_points
    
    def backproject_depth(
        self, 
        cam_idx:int, 
        depth:torch.Tensor, 
        mask:torch.Tensor=None,
        n_backprojected_points=-1,
        ):
        """_summary_

        Args:
            cam_idx (int): _description_
            depth (torch.Tensor): Has shape (H, W) or (1, H, W) or (H, W, 1).
            mask (torch.Tensor, optional): Has shape (H, W) or (1, H, W) or (H, W, 1). Defaults to None.
        """
        p3d_camera = self.p3d_cameras[cam_idx]
        image_height = self.height[cam_idx].item()
        image_width = self.width[cam_idx].item()
        cx = self.cx[cam_idx].item()
        cy = self.cy[cam_idx].item()
        
        x_tab = torch.Tensor([[i for j in range(image_width)] for i in range(image_height)]).to(depth.device)
        y_tab = torch.Tensor([[j for j in range(image_width)] for i in range(image_height)]).to(depth.device)
        ndc_x_tab = image_width / min(image_width, image_height) - (y_tab / (min(image_width, image_height) - 1)) * 2
        ndc_y_tab = image_height / min(image_width, image_height) - (x_tab / (min(image_width, image_height) - 1)) * 2

        ndc_points = torch.cat((ndc_x_tab.view(1, -1, 1).expand(1, -1, -1),
                                ndc_y_tab.view(1, -1, 1).expand(1, -1, -1),
                                depth.view(1, -1, 1)),
                                dim=-1
                                ).view(1, image_height * image_width, 3)
        if mask is not None:
            ndc_points = ndc_points[mask.view(1, -1)]  # Remove pixels outside mask
        if n_backprojected_points == -1:
            n_backprojected_points = ndc_points.shape[1]
            ndc_points_idx = torch.arange(n_backprojected_points)
        else:
            n_backprojected_points = min(n_backprojected_points, ndc_points.shape[1])
            ndc_points_idx = torch.randperm(ndc_points.shape[1])[:n_backprojected_points]
            ndc_points = ndc_points[:, ndc_points_idx]
        
        return p3d_camera.unproject_points(ndc_points, scaled_depth_input=False).view(-1, 3)
    
    
def warp_image(
    source_cameras:CamerasWrapper,
    source_idx:int,
    source_img:torch.Tensor,
    target_cameras:CamerasWrapper,
    target_idx:int,
    target_depth:torch.Tensor,
):
    """Warp an image from source_camera to target_camera.

    Args:
        source_camera (CamerasWrapper): Source cameras.
        source_idx (int): Index of the camera from which the image was taken.
        source_img (torch.Tensor): Tensor with shape (H, W, 3).
        target_camera (CamerasWrapper): Target cameras.
        target_idx (int): Index of the camera to which the image will be warped.
        target_depth (torch.Tensor): Tensor with shape (H, W).
    """
    
    height, width = target_depth.shape[:2]
    factor = -1 * min(height, width)

    pts_3d = target_cameras.backproject_depth(cam_idx=target_idx, depth=target_depth)
    pts_2d = source_cameras.p3d_cameras[source_idx].get_full_projection_transform().transform_points(pts_3d)[..., :2]

    pts_2d[..., 0] = factor / width * pts_2d[..., 0]
    pts_2d[..., 1] = factor / height * pts_2d[..., 1]

    warped_image = torch.nn.functional.grid_sample(
        source_img.permute(2, 0, 1)[None],  # (1, 3, H, W)  
        pts_2d.view(1, height, width, 2),  # (1, 1, N, 2)
        mode='nearest',
        # mode='bilinear', 
        padding_mode='zeros',
        align_corners=False
    )[0].permute(1, 2, 0)
    
    return warped_image
    
    
def get_neighbor_cameras(
    cameras1:List[GSCamera],
    cameras2:List[GSCamera],
    n_neighbors:int,
):
    """
    For each camera in cameras1, find the n_neighbors closest cameras in cameras2.
    Return a tensor with shape (len(cameras1), n_neighbors) containing the indices of the closest cameras in cameras2.
    
    Args:
        cameras1: List of GSCamera objects
        cameras2: List of GSCamera objects
        n_neighbors: Number of closest neighbors to find
        
    Returns:
        Tensor with shape (len(cameras1), n_neighbors) containing the indices of the closest cameras in cameras2.
    """
    # Get the camera centers
    camera_centers_1 = torch.stack([cam.camera_center for cam in cameras1])
    camera_centers_2 = torch.stack([cam.camera_center for cam in cameras2])
    
    # Compute the pairwise distances between the camera centers
    dists = torch.cdist(camera_centers_1, camera_centers_2, p=2)
    
    # Get the indices of the closest cameras
    closest_indices = torch.argsort(dists, dim=-1)[:, :n_neighbors]

    return closest_indices


def interpolate_between_cameras(camera1, camera2, t, use_image1_as_gt=False):
    # Move to GPU
    T1 = torch.from_numpy(camera1.T).to('cuda')
    T2 = torch.from_numpy(camera2.T).to('cuda')
    
    # Convert rotation matrices to quaternions
    quat1 = matrix_to_quaternion(torch.from_numpy(camera1.R).to('cuda'))
    quat2 = matrix_to_quaternion(torch.from_numpy(camera2.R).to('cuda'))
    
    # Interpolate between poses
    new_T = T1 + t * (T2 - T1)
    new_quat = quat1 + t * (quat2 - quat1)
    
    # Convert quaternion to rotation matrix
    new_R = quaternion_to_matrix(new_quat)
    
    # Create new camera object. 
    # Uses the intrinsic matrix of camera1
    return GSCamera(
        colmap_id=camera1.colmap_id,
        R=new_R,
        T=new_T,
        FoVx=camera1.FoVx,
        FoVy=camera1.FoVy,
        image=camera1.original_image if use_image1_as_gt else None,
        gt_alpha_mask=None,
        image_name=camera1.image_name,
        uid=camera1.uid,
        data_device=camera1.data_device,
        image_height=camera1.image_height,
        image_width=camera1.image_width,
    )
    

def get_cameras_interpolated_between_neighbors(
    cameras:List[GSCamera],
    n_neighbors_to_interpolate:int,
    n_interpolated_cameras_for_each_neighbor:int,
    debug:bool=False,
):
    # Get the neighbor camera indices (excluding the camera itself)
    neighbor_idxs = get_neighbor_cameras(
        cameras1=cameras,
        cameras2=cameras,
        n_neighbors=n_neighbors_to_interpolate + 1,
    )[..., 1:]
    
    # Build pairs of indices of neighbor cameras and remove duplicates
    camera_pairs = []
    for i in range(len(cameras)):
        for j in range(n_neighbors_to_interpolate):
            # We avoid duplicates by only adding the pair if i < neighbor_idxs[i, j].item()
            if i < neighbor_idxs[i, j].item():
                camera_pairs.append([i, neighbor_idxs[i, j].item()])    
    
    # Build the interpolated cameras
    interpolated_cameras = []
    for pair in camera_pairs:
        for t in torch.linspace(0, 1, n_interpolated_cameras_for_each_neighbor):
            interpolated_cameras.append(interpolate_between_cameras(cameras[pair[0]], cameras[pair[1]], t.item(), use_image1_as_gt=debug))
    
    return interpolated_cameras