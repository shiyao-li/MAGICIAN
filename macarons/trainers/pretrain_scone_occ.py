# Updated from former script train_proba_model_faster.py
import sys
from ..utility.scone_utils import *

dir_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(dir_path, "../../data/ShapeNetCore.v1")
weights_dir = os.path.join(dir_path, "../../weights/scone/occupancy")


def loop(params,
         batch, mesh_dict,
         scone_occ, occ_loss_fn,
         device, is_master,
         n_views_list=None
         ):
    paths = mesh_dict['path']

    pred_occs = torch.zeros(0, 1, device=device)
    truth_occs = torch.zeros(0, 1, device=device)

    base_harmonics, h_polar, h_azim = get_all_harmonics_under_degree(params.harmonic_degree,
                                                                     params.view_state_n_elev,
                                                                     params.view_state_n_azim,
                                                                     device)

    batch_size = len(paths)

    # Loading, if provided, view sequences (useful for consistent validation)
    if n_views_list is None:
        n_views = np.random.randint(params.n_view_min, params.n_view_max + 1, batch_size)
    else:
        n_views = get_validation_n_view(params, n_views_list, batch, idr_torch_rank)

    for i in range(batch_size):
        # ----------Load input mesh and ground truth data---------------------------------------------------------------

        path_i = paths[i]
        # Loading info about partial point clouds and coverages
        part_pc, _ = get_gt_partial_point_clouds(path=path_i,
                                                 normalization_factor=1.,
                                                 device=device)
        # Loading info about ground truth occupancy field
        X_world, occs = get_gt_occupancy_field(path=path_i, device=device)

        # ----------Set camera positions associated to partial point clouds---------------------------------------------

        # Positions are loaded in world coordinates
        X_cam_world, camera_dist, camera_elev, camera_azim = get_cameras_on_sphere(params, device,
                                                                                   pole_cameras=params.pole_cameras)

        # ----------Select initial observations of the object-----------------------------------------------------------

        # Select a subset of n_view cameras to compute an initial point cloud
        n_view = n_views[i]
        view_idx = torch.randperm(len(camera_elev), device=device)[:n_view]

        # Select either first camera view space, or random camera view space as prediction view space
        if params.prediction_in_random_camera_space:
            prediction_cam_idx = np.random.randint(low=0, high=len(camera_elev))
        else:
            prediction_cam_idx = view_idx[0]
        prediction_box_center = torch.Tensor([0., 0., params.camera_dist]).to(device)

        # Move camera coordinates from world space to prediction view space, and normalize them for prediction box
        prediction_R, prediction_T = look_at_view_transform(dist=camera_dist[prediction_cam_idx],
                                                            elev=camera_elev[prediction_cam_idx],
                                                            azim=camera_azim[prediction_cam_idx],
                                                            device=device)
        prediction_camera = FoVPerspectiveCameras(device=device, R=prediction_R, T=prediction_T)
        prediction_view_transform = prediction_camera.get_world_to_view_transform()

        X_cam = prediction_view_transform.transform_points(X_cam_world)
        X_cam = normalize_points_in_prediction_box(points=X_cam,
                                                   prediction_box_center=prediction_box_center,
                                                   prediction_box_diag=params.gt_max_diagonal)
        _, elev_cam, azim_cam = get_spherical_coords(X_cam)

        X_view = X_cam[view_idx]

        # ----------Capture initial observations------------------------------------------------------------------------

        # Points observed in initial views
        pc = torch.vstack([part_pc[pc_idx] for pc_idx in view_idx])

        # Downsampling partial point cloud
        pc = pc[torch.randperm(len(pc))[:n_view * params.seq_len]]

        # Move partial point cloud from world space to prediction view space, and normalize them in prediction box
        pc = prediction_view_transform.transform_points(pc)
        pc = normalize_points_in_prediction_box(points=pc,
                                                prediction_box_center=prediction_box_center,
                                                prediction_box_diag=params.gt_max_diagonal).view(1, -1, 3)

        # ----------Compute inputs to SconeOcc--------------------------------------------------------------------------

        # Sample random proxy points in space
        X_idx = torch.randperm(len(X_world))[:params.n_sample]
        X_world, occs = X_world[X_idx], occs[X_idx]

        # Move proxy points from world space to prediction view space, and normalize them in prediction box
        X = prediction_view_transform.transform_points(X_world)
        X = normalize_points_in_prediction_box(points=X,
                                               prediction_box_center=prediction_box_center,
                                               prediction_box_diag=params.gt_max_diagonal).view(1, params.n_sample, 3)

        # Compute view state vector and corresponding view harmonics
        view_state = compute_view_state(X, X_view,
                                        params.view_state_n_elev, params.view_state_n_azim)
        view_harmonics = compute_view_harmonics(view_state,
                                                base_harmonics, h_polar, h_azim,
                                                params.view_state_n_elev, params.view_state_n_azim)
        if params.no_view_harmonics:
            view_harmonics *= 0.

        # ----------Predict Occupancy Probability-----------------------------------------------------------------------
        pred_i = scone_occ(pc, X, view_harmonics).view(-1, 1)
        pred_occs = torch.vstack((pred_occs, pred_i))

        # ----------GT Occupancy Probability----------------------------------------------------------------------------
        truth_occs = torch.vstack((truth_occs, occs))

    # ----------Compute Loss--------------------------------------------------------------------------------------------
    loss = occ_loss_fn(pred_occs, truth_occs)
    if params.multiply_loss:
        loss *= params.loss_multiplication_factor

    if batch % params.empty_cache_every_n_batch == 0 and is_master:
        print("View state sum-mean:", torch.mean(torch.sum(view_state, dim=-1)))

    return loss, pred_occs, truth_occs, batch_size, n_view


def train(params,
          dataloader,
          scone_occ, occ_loss_fn,
          optimizer,
          device, is_master,
          train_losses):

    num_batches = len(dataloader)
    size = num_batches * params.total_batch_size
    train_loss = 0.

    # Preparing information model
    scone_occ.train()

    t0 = time.time()

    for batch, (mesh_dict) in enumerate(dataloader):

        loss, pred, truth, batch_size, n_screen_cameras = loop(params,
                                                               batch, mesh_dict,
                                                               scone_occ, occ_loss_fn,
                                                               device, is_master)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        if batch % params.empty_cache_every_n_batch == 0 and is_master and params.check_gradients:
            check_gradients(scone_occ)
        optimizer.step()

        train_loss += loss.detach()
        if params.multiply_loss:
            train_loss /= params.loss_multiplication_factor

        if batch % params.empty_cache_every_n_batch == 0:

            # loss = reduce_tensor(loss)
            if params.ddp or params.jz:
                loss = reduce_tensor(loss, world_size=params.WORLD_SIZE)
            loss = to_python_float(loss)

            current = batch * batch_size # * idr_torch.size
            if params.ddp or params.jz:
                current *= params.WORLD_SIZE

            truth_norm = to_python_float(torch.linalg.norm(truth.detach()))
            pred_norm = to_python_float(torch.linalg.norm(pred.detach()))

            # torch.cuda.synchronize()

            if is_master:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]",
                      "computed in", (time.time() - t0) / 60., "minutes.")
                print(">>>Prediction shape:", pred.shape,
                      "\n>>>Truth norm:", truth_norm, ">>>Prediction norm:", pred_norm,
                      "\nNumber of cameras:", n_screen_cameras)
            # print("Harmonics:\n", info_harmonics_i)
            # TO REMOVE
            torch.cuda.empty_cache()

        if batch % params.empty_cache_every_n_batch == 0:
            t0 = time.time()

    # train_loss = reduce_tensor(train_loss)
    if params.ddp or params.jz:
        train_loss = reduce_tensor(train_loss, world_size=params.WORLD_SIZE)
    train_loss = to_python_float(train_loss)
    train_loss /= num_batches
    train_losses.append(train_loss)


def validation(params,
               dataloader,
               occ_model, occ_loss_fn,
               device, is_master,
               val_losses):

    num_batches = len(dataloader)
    size = num_batches * params.total_batch_size
    val_loss = 0.

    # Preparing information model
    occ_model.eval()

    t0 = time.time()

    n_views_list = get_validation_n_views_list(params, dataloader)

    for batch, (mesh_dict) in enumerate(dataloader):
        with torch.no_grad():
            loss, pred, truth, batch_size, _ = loop(params,
                                                    batch, mesh_dict,
                                                    occ_model, occ_loss_fn,
                                                    device, is_master,
                                                    n_views_list=n_views_list)

        val_loss += loss.detach()
        if params.multiply_loss:
            val_loss /= params.loss_multiplication_factor

        if batch % params.empty_cache_every_n_batch == 0:
            torch.cuda.empty_cache()

    # val_loss = reduce_tensor(val_loss)
    if params.ddp or params.jz:
        val_loss = reduce_tensor(val_loss, world_size=params.WORLD_SIZE)

    val_loss = to_python_float(val_loss)
    val_loss /= num_batches
    val_losses.append(val_loss)

    if is_master:
        print(f"Validation Error: \n Avg loss: {val_loss:>8f} \n")


def run_training(ddp_rank=None, params=None):
    # Set device
    device = setup_device(params, ddp_rank)

    batch_size = params.batch_size
    total_batch_size = params.total_batch_size

    if params.ddp:
        world_size = params.WORLD_SIZE
        rank = ddp_rank
        is_master = rank == 0
    elif params.jz:
        world_size = idr_torch_size
        rank = idr_torch_rank
        is_master = rank == 0
    else:
        world_size, rank = None, None
        is_master = True

    # Create dataloader
    train_dataloader, val_dataloader, _ = get_shapenet_dataloader(batch_size=params.batch_size,
                                                                  ddp=params.ddp, jz=params.jz,
                                                                  world_size=world_size, ddp_rank=rank,
                                                                  load_obj=False,
                                                                  data_path=params.data_path)


    # Initialize or Load models
    if params.no_local_features:
        raise NameError("no_local_features mode is not implemented yet.")
    else:
        scone_occ = SconeOcc().to(device)

    best_train_loss = 1000
    epochs_without_improvement = 0

    # Initialize information model (and DDP wrap if needed)
    scone_occ, optimizer, opt_name, start_epoch, best_loss, best_train_loss = initialize_scone_occ(params=params,
                                                                                                   scone_occ=scone_occ,
                                                                                                   device=device,
                                                                                                   torch_seed=params.torch_seed,
                                                                                                   load_pretrained_weights=not params.start_from_scratch,
                                                                                                   pretrained_weights_name=params.pretrained_weights_name,
                                                                                                   ddp_rank=rank,
                                                                                                   return_best_train_loss=True)

    learning_rate = params.learning_rate
    if not params.start_from_scratch:
        for lr_epoch in params.lr_epochs:
            if lr_epoch < start_epoch:
                print("Multiplying learning rate by", params.lr_factor)
                learning_rate *= params.lr_factor

    # Set loss function
    occ_loss_fn = get_occ_loss_fn(params)

    if is_master:
        print("Model name:", params.scone_occ_model_name, "\nArchitecture:\n")
        print(scone_occ)
        print("Model name:", params.scone_occ_model_name)
        print("Numbers of trainable parameters:", count_parameters(scone_occ))
        print("Using", opt_name, "optimizer.")

        if params.training_loss == "cross_entropy":
            print("Using soft cross entropy loss.")
        elif params.training_loss == "mse":
            print("Using MSE loss.")

        print("Using", params.n_camera, "uniformly sampled camera position per mesh.")

        print("Training data:", len(train_dataloader), "batches.")
        print("Validation data:", len(val_dataloader), "batches.")
        print("Batch size:", params.total_batch_size)
        print("Batch size per GPU:", params.batch_size)

        print("Best train loss:", best_train_loss)

    # Begin training process
    train_losses = []
    val_losses = []
    val_coverages = []

    t0 = time.time()
    for t_e in range(params.epochs):
        t = start_epoch + t_e
        if is_master:
            print(f"Epoch {t + 1}\n-------------------------------")
        torch.cuda.empty_cache()

        if params.schedule_learning_rate:
            if t in params.lr_epochs:
                print("Multiplying learning rate by", params.lr_factor)
                learning_rate *= params.lr_factor

        update_learning_rate(params, optimizer, learning_rate)

        print("Max learning rate set to", learning_rate)
        print("Current learning rate set to", optimizer._rate)

        train_dataloader.sampler.set_epoch(t)

        scone_occ.train()
        train(params,
              train_dataloader,
              scone_occ, occ_loss_fn,
              optimizer,
              device, is_master,
              train_losses)

        current_loss = train_losses[-1]

        if is_master:
            print("Training done for epoch", t + 1, ".")
            model_save_path = "unvalidated_" + params.scone_occ_model_name + ".pth"
            model_save_path = os.path.join(weights_dir, model_save_path)
            torch.save({
                'epoch': t + 1,
                'model_state_dict': scone_occ.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, model_save_path)

            if current_loss < best_train_loss:
                best_model_save_path = "best_unval_" + params.scone_occ_model_name + ".pth"
                best_model_save_path = os.path.join(weights_dir, best_model_save_path)
                torch.save({
                    'epoch': t + 1,
                    'model_state_dict': scone_occ.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': current_loss,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                }, best_model_save_path)
                best_train_loss = current_loss
                print("Best model on training set saved with loss " + str(current_loss) + " .\n")

        torch.cuda.empty_cache()

        if is_master:
            print("Beginning evaluation on validation dataset...")
        # val_dataloader.sampler.set_epoch(t)

        scone_occ.eval()
        validation(params,
                   val_dataloader,
                   scone_occ, occ_loss_fn,
                   device, is_master,
                   val_losses
                   )

        current_val_loss = val_losses[-1]
        if current_val_loss < best_loss:
            best_model_save_path = "validated_" + params.scone_occ_model_name + ".pth"
            best_model_save_path = os.path.join(weights_dir, best_model_save_path)
            if is_master:
                torch.save({
                    'epoch': t + 1,
                    'model_state_dict': scone_occ.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': current_val_loss,
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }, best_model_save_path)
                print("Model saved with loss " + str(current_val_loss) + " .\n")
            best_loss = val_losses[-1]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Save data about losses
        if is_master:
            losses_data = {}
            losses_data['train_loss'] = train_losses
            losses_data['val_loss'] = val_losses
            json_name = "losses_data_" + params.scone_occ_model_name + ".json"
            with open(json_name, 'w') as outfile:
                json.dump(losses_data, outfile)
            print("Saved data about losses in", json_name, ".")

    if is_master:
        print("Done in", (time.time() - t0) / 3600., "hours!")

        # Save data about losses
        losses_data = {}
        losses_data['train_loss'] = train_losses
        losses_data['val_loss'] = val_losses
        json_name = "losses_data_" + params.scone_occ_model_name + ".json"
        with open(json_name, 'w') as outfile:
            json.dump(losses_data, outfile)
        print("Saved data about losses in", json_name, ".")

    if params.ddp or params.jz:
        cleanup()
