# Updated from former script train_discoveries_faster.py
import sys
from ..utility.scone_utils import *

dir_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(dir_path, "../../data/ShapeNetCore.v1")
weights_dir = os.path.join(dir_path, "../../weights/scone/coverage_gain")


def loop(params,
         batch, mesh_dict,
         scone_occ, scone_vis, cov_loss_fn,
         device, is_master,
         n_views_list=None,
         optimal_sequences=None
         ):
    paths = mesh_dict['path']

    cov_pred = torch.zeros(0, params.n_camera, 1, device=device)
    cov_truth = torch.zeros(0, params.n_camera, 1, device=device)

    info_harmonics_i = None

    base_harmonics, h_polar, h_azim = get_all_harmonics_under_degree(params.harmonic_degree,
                                                                     params.view_state_n_elev,
                                                                     params.view_state_n_azim,
                                                                     device)

    batch_size = len(paths)

    if n_views_list is None:
        n_views = np.random.randint(params.n_view_min, params.n_view_max + 1, batch_size)
    else:
        n_views = get_validation_n_view(params, n_views_list, batch, idr_torch_rank)

    if batch == 0 and is_master:
        print("First batch:", mesh_dict['path'])

    for i in range(batch_size):
        # ----------Load input mesh and ground truth data---------------------------------------------------------------

        path_i = paths[i]

        # Loading info about partial point clouds and coverages
        part_pc, coverage = get_gt_partial_point_clouds(path=path_i,
                                                        normalization_factor=1.,
                                                        device=device)

        # Loading info about ground truth surface
        # gt_surface, surface_epsilon = get_gt_surface(params=params,
        #                                              path=path_i,
        #                                              normalization_factor=1.,
        #                                              device=device)

        # Initial dense sampling
        X_world = sample_X_in_box(x_range=params.gt_max_diagonal, n_sample=params.n_proxy_points, device=device)

        # ----------Set camera candidates for coverage prediction-------------------------------------------------------
        X_cam_world, camera_dist, camera_elev, camera_azim = get_cameras_on_sphere(params, device,
                                                                                   pole_cameras=params.pole_cameras)

        # ----------Select initial observations of the object-----------------------------------------------------------

        # Select a subset of n_view cameras to compute an initial point cloud
        n_view = n_views[i]
        if optimal_sequences is None:
            view_idx = torch.randperm(len(camera_elev), device=device)[:n_view]
        else:
            optimal_seq, _ = get_optimal_sequence(optimal_sequences, path_i, n_view)
            view_idx = optimal_seq.to(device)

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
        X_cam = X_cam.view(1, params.n_camera, 3)

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

        # ----------Compute inputs to SconeVis-----------------------------------------------

        # Sample random proxy points in space
        X_idx = torch.randperm(len(X_world))[:params.n_proxy_points]
        X_world = X_world[X_idx]

        # Move proxy points from world space to prediction view space, and normalize them in prediction box
        X = prediction_view_transform.transform_points(X_world)
        X = normalize_points_in_prediction_box(points=X,
                                               prediction_box_center=prediction_box_center,
                                               prediction_box_diag=params.gt_max_diagonal
                                               )

        # Filter Proxy Points using pc shape from view cameras
        R_view, T_view = look_at_view_transform(eye=X_view,
                                                at=torch.zeros_like(X_view),
                                                device=device)
        view_cameras = FoVPerspectiveCameras(R=R_view, T=T_view, zfar=1000, device=device)
        X, _ = filter_proxy_points(view_cameras, X, pc.view(-1, 3), filter_tol=params.filter_tol)
        X = X.view(1, X.shape[0], 3)

        # Compute view state vector and corresponding view harmonics
        view_state = compute_view_state(X, X_view,
                                        params.view_state_n_elev, params.view_state_n_azim)
        view_harmonics = compute_view_harmonics(view_state,
                                                base_harmonics, h_polar, h_azim,
                                                params.view_state_n_elev, params.view_state_n_azim)
        occ_view_harmonics = 0. + view_harmonics
        if params.occ_no_view_harmonics:
            occ_view_harmonics *= 0.
        if params.no_view_harmonics:
            view_harmonics *= 0.

        # Compute occupancy probabilities
        with torch.no_grad():
            occ_prob_i = compute_occupancy_probability(scone_occ=scone_occ,
                                                       pc=pc,
                                                       X=X,
                                                       view_harmonics=occ_view_harmonics,
                                                       max_points_per_pass=params.max_points_per_scone_occ_pass
                                                       ).view(-1, 1)

        proxy_points, view_harmonics, sample_idx = sample_proxy_points(X[0], occ_prob_i, view_harmonics.squeeze(dim=0),
                                                                       n_sample=params.seq_len,
                                                                       min_occ=params.min_occ_for_proxy_points,
                                                                       use_occ_to_sample=params.use_occ_to_sample_proxy_points,
                                                                       return_index=True)

        proxy_points = torch.unsqueeze(proxy_points, dim=0)
        view_harmonics = torch.unsqueeze(view_harmonics, dim=0)

        # ----------Predict Coverage Gains------------------------------------------------------------------------------
        visibility_gain_harmonics = scone_vis(proxy_points, view_harmonics=view_harmonics)
        if params.true_monte_carlo_sampling:
            proxy_points = torch.unsqueeze(proxy_points[0][sample_idx], dim=0)
            visibility_gain_harmonics = torch.unsqueeze(visibility_gain_harmonics[0][sample_idx], dim=0)

        if params.ddp or params.jz:
            cov_pred_i = scone_vis.module.compute_coverage_gain(proxy_points,
                                                                visibility_gain_harmonics,
                                                                X_cam)
        else:
            cov_pred_i = scone_vis.compute_coverage_gain(proxy_points,
                                                         visibility_gain_harmonics,
                                                         X_cam)

        cov_pred = torch.vstack((cov_pred, cov_pred_i.view(1, -1, 1)))

        # ----------Compute ground truth information scores----------
        cov_truth_i = compute_gt_coverage_gain_from_precomputed_matrices(coverage=coverage,
                                                                         initial_cam_idx=view_idx)
        cov_truth = torch.vstack((cov_truth, cov_truth_i.view(1, -1, 1)))

    # ----------Compute loss----------
    # cov_pred = cov_pred.view(-1, params.n_camera, 1)
    # cov_truth = cov_truth.view(-1, params.n_camera, 1)
    loss = cov_loss_fn(cov_pred, cov_truth)

    if batch % params.empty_cache_every_n_batch == 0 and is_master:
        print("View state sum-mean:", torch.mean(torch.sum(view_state, dim=-1)))
        print("Point cloud features shape:", proxy_points.shape)
        # print("Surface epsilon:", params.surface_epsilon)

    return loss, cov_pred, cov_truth, batch_size, n_view


def train(params,
          dataloader,
          scone_occ,
          scone_vis, cov_loss_fn,
          optimizer,
          device, is_master,
          train_losses):

    num_batches = len(dataloader)
    size = num_batches * params.total_batch_size
    train_loss = 0.

    # Preparing information model
    scone_vis.train()

    t0 = time.time()

    for batch, (mesh_dict) in enumerate(dataloader):

        loss, cov_pred, cov_truth, batch_size, n_view = loop(params,
                                                             batch, mesh_dict,
                                                             scone_occ, scone_vis, cov_loss_fn,
                                                             device, is_master,
                                                             n_views_list=None,
                                                             optimal_sequences=None)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
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

            truth_norm = to_python_float(torch.linalg.norm(cov_truth.detach()))
            pred_norm = to_python_float(torch.linalg.norm(cov_pred.detach()))

            # torch.cuda.synchronize()

            if is_master:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]",
                      "computed in", (time.time() - t0) / 60., "minutes.")
                print(">>>Prediction shape:", cov_pred.shape,
                      "\n>>>Truth norm:", truth_norm, ">>>Prediction norm:", pred_norm,
                      "\nNumber of cameras:", n_view, "+ 1.")
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
               scone_occ,
               scone_vis, cov_loss_fn,
               device, is_master,
               val_losses,
               nbv_validation=False,
               val_coverages=None):

    num_batches = len(dataloader)
    size = num_batches * params.total_batch_size
    val_loss = 0.
    val_coverage = 0.

    # Preparing information model
    scone_vis.eval()

    t0 = time.time()

    n_views_list = get_validation_n_views_list(params, dataloader)
    optimal_sequences = get_validation_optimal_sequences(jz=params.jz, device=device)

    for batch, (mesh_dict) in enumerate(dataloader):
        with torch.no_grad():
            loss, cov_pred, cov_truth, batch_size, _ = loop(params,
                                                            batch, mesh_dict,
                                                            scone_occ, scone_vis, cov_loss_fn,
                                                            device, is_master,
                                                            n_views_list=n_views_list,
                                                            optimal_sequences=optimal_sequences)

        val_loss += loss.detach()
        if params.multiply_loss:
            val_loss /= params.loss_multiplication_factor

        if nbv_validation:
            with torch.no_grad():
                info_pred_scores = cov_pred.view(batch_size, params.n_camera, 1)
                info_pred_scores = torch.squeeze(info_pred_scores, dim=-1)

                info_truth_scores = cov_truth.view(batch_size, params.n_camera, 1)
                info_truth_scores = torch.squeeze(info_truth_scores, dim=-1)

                max_info_preds, max_info_idx = torch.max(info_pred_scores,
                                                         dim=1)

                max_idx = max_info_idx.view(batch_size, 1)
                true_max_coverages = torch.gather(info_truth_scores,
                                                  dim=1,
                                                  index=max_idx)

                val_coverage += torch.sum(true_max_coverages).detach() / batch_size

        if batch % params.empty_cache_every_n_batch == 0:
            torch.cuda.empty_cache()

    # val_loss = reduce_tensor(val_loss)
    if params.ddp or params.jz:
        val_loss = reduce_tensor(val_loss, world_size=params.WORLD_SIZE)
        if nbv_validation:
            val_coverage = reduce_tensor(val_coverage, world_size=params.WORLD_SIZE)

    val_loss = to_python_float(val_loss)
    val_loss /= num_batches
    val_losses.append(val_loss)

    if nbv_validation:
        val_coverage = to_python_float(val_coverage)
        val_coverage /= num_batches
        if val_coverages is None:
            raise NameError("Variable val_coverages is set to None.")
        else:
            val_coverages.append(val_coverage)

    if is_master:
        print(f"Validation Error: \n Avg loss: {val_loss:>8f} \n")
        if nbv_validation:
            print(f"Avg nbv coverage: {val_coverage:>8f} \n")


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
    scone_occ = load_scone_occ(params, params.scone_occ_model_name, ddp_model=True, device=device)
    scone_occ.eval()

    best_train_loss = 1000
    epochs_without_improvement = 0

    scone_vis = SconeVis(use_sigmoid=params.use_sigmoid).to(device)
    scone_vis, optimizer, opt_name, start_epoch, best_loss, best_coverage, best_train_loss = initialize_scone_vis(params=params,
        scone_vis=scone_vis,
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
    cov_loss_fn = get_cov_loss_fn(params)

    if is_master:
        print("Model name:", params.scone_vis_model_name, "\nArchitecture:\n")
        print(scone_vis)
        print("Model name:", params.scone_vis_model_name)
        print("Numbers of trainable parameters:", count_parameters(scone_vis))
        print("Using", opt_name, "optimizer.")
        print("Using occupancy model", params.scone_occ_model_name)

        if params.training_loss == "kl_divergence":
            print("Using softmax + KL Divergence loss.")
        elif params.training_loss == "mse":
            print("Using MSE loss.")

        print("Using", params.n_camera, "uniformly sampled camera position per mesh.")

        print("Training data:", len(train_dataloader), "batches.")
        print("Validation data:", len(val_dataloader), "batches.")
        print("Batch size:", params.total_batch_size)
        print("Batch size per GPU:", params.batch_size)

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

        # Update learning rate
        if params.schedule_learning_rate:
            if t in params.lr_epochs:
                print("Multiplying learning rate by", params.lr_factor)
                learning_rate *= params.lr_factor

        update_learning_rate(params, optimizer, learning_rate)
        print("Max learning rate set to", learning_rate)
        print("Current learning rate set to", optimizer._rate)

        train_dataloader.sampler.set_epoch(t)

        scone_vis.train()
        train(params,
              train_dataloader,
              scone_occ,
              scone_vis, cov_loss_fn,
              optimizer,
              device, is_master,
              train_losses
              )

        current_loss = train_losses[-1]

        if is_master:
            print("Training done for epoch", t + 1, ".")
            model_save_path = "unvalidated_" + params.scone_vis_model_name + ".pth"
            model_save_path = os.path.join(weights_dir, model_save_path)
            torch.save({
                'epoch': t + 1,
                'model_state_dict': scone_vis.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_loss,
                # 'coverage': current_val_coverage,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, model_save_path)

            if current_loss < best_train_loss:
                best_model_save_path = "best_unval_" + params.scone_vis_model_name + ".pth"
                best_model_save_path = os.path.join(weights_dir, best_model_save_path)
                torch.save({
                    'epoch': t + 1,
                    'model_state_dict': scone_vis.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': current_loss,
                    # 'coverage': current_val_coverage,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                }, best_model_save_path)
                best_train_loss = current_loss
                print("Best model on training set saved with loss " + str(current_loss) + " .\n")

        torch.cuda.empty_cache()

        if is_master:
            print("Beginning evaluation on validation dataset...")
        # val_dataloader.sampler.set_epoch(t)

        scone_vis.eval()
        validation(params,
                   val_dataloader,
                   scone_occ,
                   scone_vis, cov_loss_fn,
                   device, is_master,
                   val_losses,
                   nbv_validation=params.nbv_validation,
                   val_coverages=val_coverages
                   )

        current_val_loss = val_losses[-1]
        current_val_coverage = val_coverages[-1]
        if current_val_loss < best_loss:
            best_model_save_path = "validated_" + params.scone_vis_model_name + ".pth"
            best_model_save_path = os.path.join(weights_dir, best_model_save_path)
            if is_master:
                torch.save({
                    'epoch': t + 1,
                    'model_state_dict': scone_vis.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': current_val_loss,
                    'coverage': current_val_coverage,
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }, best_model_save_path)
                print("Model saved with loss " + str(current_val_loss) + " .\n")
            best_loss = val_losses[-1]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if is_master and current_val_coverage > best_coverage:
            best_model_save_path = "coverage_validated_" + params.scone_vis_model_name + ".pth"
            best_model_save_path = os.path.join(weights_dir, best_model_save_path)
            torch.save({
                'epoch': t + 1,
                'model_state_dict': scone_vis.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_val_loss,
                'coverage': current_val_coverage,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, best_model_save_path)
            print("Model saved with coverage " + str(current_val_coverage) + " .\n")
            best_coverage = val_coverages[-1]

        # Save data about losses
        if is_master:
            losses_data = {}
            losses_data['train_loss'] = train_losses
            losses_data['val_loss'] = val_losses
            json_name = "losses_data_" + params.scone_vis_model_name + ".json"
            with open(json_name, 'w') as outfile:
                json.dump(losses_data, outfile)
            print("Saved data about losses in", json_name, ".")

    if is_master:
        print("Done in", (time.time() - t0) / 3600., "hours!")

        # Save data about losses
        losses_data = {}
        losses_data['train_loss'] = train_losses
        losses_data['val_loss'] = val_losses
        json_name = "losses_data_" + params.scone_vis_model_name + ".json"
        with open(json_name, 'w') as outfile:
            json.dump(losses_data, outfile)
        print("Saved data about losses in", json_name, ".")

    if params.ddp or params.jz:
        cleanup()
