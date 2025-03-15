import os 
import torch
from torch.utils.data import DataLoader

from network.model import GauPCRender
from network.dataset import GaussianDataset, GaussianPatchDataset
from progress import get_progress

def train(cmd_args, logging):
    logging.info("Training GauPCRender")

    max_epoch = cmd_args.max_epoch
    batch_size = cmd_args.batch_size
    lr = cmd_args.lr

    if cmd_args.scene_cate:
        batch_size = 1

    train_name = cmd_args.train_name
    save_path = os.path.join(cmd_args.save_path, train_name)

    if not cmd_args.patch:
        logging.info(f'Loading dataset')
        dataset = GaussianDataset(cmd_args, 'train')
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, collate_fn=GaussianDataset.collet_fn)
    else:
        logging.info(f'Loading dataset (patch)')
        dataset = GaussianPatchDataset(cmd_args, 'train')
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, collate_fn=GaussianPatchDataset.collet_fn)
        

    logging.info(f'Loading GauPCRender')
    model = GauPCRender(cmd_args, patch=cmd_args.patch)
    if cmd_args.restore is not None:
        weight_path = os.path.join(save_path, f'weight_{int(cmd_args.restore)}.pkl')
        logging.info(f'Load weight from: {weight_path}')
        model.load_state_dict(torch.load(weight_path))
    elif cmd_args.scene_cate:
        if cmd_args.scene_train_name is None or cmd_args.scene_restore is None:
            assert False, 'You must provide the weight when training a new model for a scene category. Please provide the weight of a Patch Model using --scene_train_name and --scene_restore.'
        else:
            weight_path = os.path.join(cmd_args.save_path, cmd_args.scene_train_name, f'weight_{int(cmd_args.scene_restore)}.pkl')
            logging.info(f'Load weight from: {weight_path}')
            model.load_state_dict(torch.load(weight_path))
    model.train()
    model.cuda()

    if not cmd_args.scene_cate and cmd_args.patch:
        entire_model = GauPCRender(cmd_args)
        weight_path = os.path.join(cmd_args.save_path, cmd_args.entire_train_name, f'weight_{int(cmd_args.entire_restore)}.pkl')
        logging.info(f'Load entire weight from: {weight_path}')
        entire_model.load_state_dict(torch.load(weight_path))
        entire_model.eval()
        entire_model.cuda()

    logging.info('Loading optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    print('<----------CONFIG---------->')
    print(f'Train name: {train_name}')
    print(f'Category: {cmd_args.cate}')
    print(f'Save path: {save_path}')
    print(f'Batch size: {batch_size}')
    print(f'Learning rate: {lr}')
    print(f'Max epoch: {max_epoch}')
    print(f'Restore: {cmd_args.restore}')
    print(f'Save epochs: {cmd_args.save_epochs}')
    print(f'Image number: {cmd_args.image_number}')
    print(f'Point number: {cmd_args.point_number}')
    print(f'Split number: {cmd_args.split_number}')
    print(f'Sh degree: {cmd_args.sh}')
    print(f'Patch: {cmd_args.patch}')
    if cmd_args.patch:
        if not cmd_args.scene_cate:
            print(f'Entire train name: {cmd_args.entire_train_name}')
            print(f'Entire restore: {cmd_args.entire_restore}')
        else:
            print(f'Scene train name: {cmd_args.scene_train_name}')
            print(f'Scene restore: {cmd_args.scene_restore}')
    
    if cmd_args.restore is not None:
        start_epoch = cmd_args.restore
    else:
        start_epoch = 0

    epoch_step_num = len(dataset) // batch_size
    totoal_step_num = (max_epoch-start_epoch) * epoch_step_num 
    print(f'Training, max epoch: {max_epoch}, total step: {totoal_step_num}')
    print('<----------CONFIG---------->')

    # confrom the config
    if not cmd_args.skip_check:
        if input('Do you want to continue? (y/n)').lower() != 'y':
            exit()

    progress = get_progress()
    epoch_p = progress.add_task('EPOCH', total=max_epoch, loss='')
    step_p = progress.add_task('STEP', total=epoch_step_num, loss='')
    with progress:
        for i in range(start_epoch, max_epoch):
            progress.reset(step_p)
            for j, data in enumerate(dataloader):
                data = dataset.to_cuda(data)
                if not cmd_args.scene_cate:
                    # training object category
                    if not cmd_args.patch:
                        # Entire Model
                        output = model(data)
                        loss = model.loss(output, data)[0]
                    else:
                        # Patch Model
                        patch_output = model(data[:3])
                        with torch.no_grad():
                            entire_output = entire_model(data[3:6])
                        loss = model.loss_patch(patch_output, entire_output, data)[0]
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    # Training scene category
                    __b = data[0][0].shape[0]
                    # if cmd_args.show_shape_log:
                    # print(data[0][0].shape)
                    if __b == 1:
                        output = model([data[0][0].repeat(2,1,1), data[1][0].repeat(2,1,1), data[2][0].repeat(2,1,1)])
                        output = [x[:1] for x in output]
                    elif __b > 5:
                        output_1 = model([data[0][0][:4], data[1][0][:4], data[2][0][:4]])
                        with torch.no_grad():
                            output_2 = model([data[0][0][4:], data[1][0][4:], data[2][0][4:]])
                            # output_2 = batch_forward(pgsn, [data[0][0][4:], data[1][0][4:], data[2][0][4:]])
                        output = []
                        for k in range(len(output_1)):
                            output.append(torch.cat((output_1[k], output_2[k]), dim=0))
                    else:
                        output = model([data[0][0], data[1][0], data[2][0]])
                    xyz, p_rgb, p_o, p_sxyz, p_q, p_sh = output 

                    anchor = data[6][0]
                    anchor = torch.unsqueeze(anchor, 1)
                    xyz    = xyz + anchor
                    xyz    = torch.reshape(xyz,    [1, -1, 3])
                    p_rgb  = torch.reshape(p_rgb,  [1, -1, 3])
                    p_o    = torch.reshape(p_o,    [1, -1, 1])
                    p_sxyz = torch.reshape(p_sxyz, [1, -1, 2])
                    p_q    = torch.reshape(p_q,    [1, -1, 4])
                    p_sh   = torch.reshape(p_sh,   [1, -1, p_sh.shape[2]])
                
                    output = xyz, p_rgb, p_o, p_sxyz, p_q, p_sh
                    loss = model.loss(output, data, epoch=i, step=j)[0]
                    optimizer.zero_grad()
                    if __b > 5:
                        loss.backward()
                        max_norm = 1.0
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                        optimizer.step()
                
                # progress.advance(step_p)
                progress.update(step_p, advance=1, loss=f'Loss: {loss.item():.4f}', refresh=True)
                progress.advance(epoch_p, 1/epoch_step_num)

            if i+1 in cmd_args.save_epochs:
                # save weights
                os.makedirs(save_path, exist_ok=True)
                weight_path = os.path.join(save_path, f'weight_{i+1}.pkl')
                torch.save(model.state_dict(), weight_path)
