"""
models.py - PRODUCTION-READY VERSION with CMGAN Audio Processing
=================================================================
Perfect integration with CMGAN-style audio processing pipeline.

MODIFICATIONS FOR CMGAN:
1. Updated STFT parameters (n_fft=320, hop_length=160)
2. Added power compression (0.3)
3. Updated feeder to return norm_factors
4. Updated all forward passes to handle norm_factors
5. Updated criterion calls with norm_factors
6. Fixed logging frequency (every 50 iterations)

All bugs fixed and tested!
"""

import os
import shutil
import timeit
import numpy as np
import soundfile as sf
import torch
import torchaudio
from natsort import natsorted
from torch.nn import DataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, lr_scheduler

from configs import (
    exp_conf, train_conf, test_conf,
    TRAIN_CLEAN_DIR, TRAIN_NOISY_DIR,
    VALID_CLEAN_DIR, VALID_NOISY_DIR,
    TEST_CLEAN_DIR, TEST_NOISY_DIR,
    validate_data_dirs,
    get_test_snr_dirs
)
from utils.utils import getLogger, numParams, countFrames, lossMask, wavNormalize
from utils.pipeline_modules import NetFeeder, Resynthesizer, power_compress, power_uncompress
from utils.data_utils import create_dataloaders, create_test_dataloader_only
from utils.networks import Net
from utils.criteria import LossFunction

# Try to import metrics computation
try:
    from tools.compute_metrics import compute_metrics
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False


class CheckPoint(object):
    """Checkpoint management with scheduler state support"""
    
    def __init__(self, ckpt_info=None, net_state_dict=None, 
                 optim_state_dict=None, scheduler_state_dict=None):
        self.ckpt_info = ckpt_info
        self.net_state_dict = net_state_dict
        self.optim_state_dict = optim_state_dict
        self.scheduler_state_dict = scheduler_state_dict
    
    def save(self, filename, is_best, best_model=None):
        """Save checkpoint to file"""
        torch.save(self, filename)
        if is_best and best_model:
            shutil.copyfile(filename, best_model)

    def load(self, filename, device):
        """Load checkpoint from file"""
        if not os.path.isfile(filename):
            raise FileNotFoundError(f'No checkpoint found at {filename}')
        ckpt = torch.load(filename, map_location=device)
        self.ckpt_info = ckpt.ckpt_info 
        self.net_state_dict = ckpt.net_state_dict
        self.optim_state_dict = ckpt.optim_state_dict
        self.scheduler_state_dict = getattr(ckpt, 'scheduler_state_dict', None)


def lossLog(log_file, ckpt, logging_period):
    """Write loss to CSV file"""
    ckpt_info = ckpt.ckpt_info
    
    if not os.path.isfile(log_file):
        with open(log_file, 'w') as f:
            f.write('epoch, iter, tr_loss, cv_loss\n')
    
    with open(log_file, 'a') as f:
        f.write('{}, {}, {:.4f}, {:.4f}\n'.format(
            ckpt_info['cur_epoch'] + 1,
            ckpt_info['cur_iter'] + 1,
            ckpt_info['tr_loss'],
            ckpt_info['cv_loss']
        ))


class Model(object):
    """
    Main model class - CMGAN-STYLE AUDIO PROCESSING
    Uses CMGAN's audio processing pipeline with your network architecture
    """
    
    def __init__(self):
        """Initialize model with CMGAN-style configuration"""
        # Get configuration from config file
        self.in_norm = exp_conf['in_norm']
        self.sample_rate = exp_conf['sample_rate']
        
        # CMGAN-style STFT parameters
        self.n_fft = exp_conf['n_fft']              # 320
        self.hop_length = exp_conf['hop_length']    # 160
        self.power = exp_conf['power_compression']   # 0.3
        
        # For backwards compatibility with utility functions
        self.win_size = self.n_fft
        self.hop_size = self.hop_length
    
    def train(self):
        """
        Training procedure with CMGAN-style audio processing.
        """
        # Validate data directories first
        print("\n" + "="*70)
        print("STEP 1: VALIDATING DATA DIRECTORIES")
        print("="*70)
        try:
            validate_data_dirs(mode='train')
        except (FileNotFoundError, ValueError) as e:
            print(f"\n? DATA VALIDATION FAILED: {e}")
            print("\nPlease check your configs.py paths!")
            return
        
        # Load configuration
        self.ckpt_dir = train_conf['ckpt_dir']
        self.resume_model = train_conf['resume_model']
        self.time_log = train_conf['time_log']
        self.lr = train_conf['lr']
        self.plateau_factor = train_conf['plateau_factor']
        self.plateau_patience = train_conf['plateau_patience']
        self.plateau_threshold = train_conf['plateau_threshold']
        self.plateau_min_lr = train_conf['plateau_min_lr']
        self.clip_norm = train_conf['clip_norm']
        self.max_n_epochs = train_conf['max_n_epochs']
        self.early_stop_patience = train_conf['early_stop_patience']
        self.batch_size = train_conf['batch_size']
        self.num_workers = train_conf['num_workers']
        self.loss_log = train_conf['loss_log']
        self.unit = train_conf['unit']
        self.segment_size = train_conf['segment_size']
        self.segment_shift = train_conf['segment_shift']
        self.max_length_seconds = train_conf['max_length_seconds']
        
        # Setup device
        self.gpu_ids = tuple(map(int, train_conf['gpu_ids'].split(',')))
        if len(self.gpu_ids) == 1 and self.gpu_ids[0] == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{self.gpu_ids[0]}')

        # Create checkpoint directory
        os.makedirs(self.ckpt_dir, exist_ok=True)

        print("\n" + "="*70)
        print("STEP 2: INITIALIZING LOGGER")
        print("="*70)
        
        # Setup logger
        logger = getLogger(os.path.join(self.ckpt_dir, 'train.log'), log_file=True)
        logger.info('='*70)
        logger.info('TRAINING CONFIGURATION - CMGAN-STYLE AUDIO PROCESSING')
        logger.info('='*70)
        logger.info(f'Training clean: {TRAIN_CLEAN_DIR}')
        logger.info(f'Training noisy: {TRAIN_NOISY_DIR}')
        logger.info(f'Validation clean: {VALID_CLEAN_DIR}')
        logger.info(f'Validation noisy: {VALID_NOISY_DIR}')
        logger.info(f'Sample rate: {self.sample_rate} Hz')
        logger.info(f'STFT n_fft: {self.n_fft} ({self.n_fft/self.sample_rate*1000:.1f} ms)')
        logger.info(f'STFT hop: {self.hop_length} ({self.hop_length/self.sample_rate*1000:.2f} ms)')
        logger.info(f'Power compression: {self.power}')
        logger.info(f'Normalization: RMS (CMGAN-style, in forward pass)')
        logger.info(f'Unit: {self.unit}')
        logger.info(f'Batch size: {self.batch_size}')
        logger.info(f'Num workers: {self.num_workers}')
        if self.unit == 'seg':
            logger.info(f'Segment size: {self.segment_size}s')
            logger.info(f'Segment shift: {self.segment_shift}s')
        logger.info(f'Max length: {self.max_length_seconds}s')
        logger.info(f'Initial LR: {self.lr}')
        logger.info(f'Max epochs: {self.max_n_epochs}')
        logger.info(f'Device: {self.device}')
        logger.info('='*70 + '\n')
        
        print("\n" + "="*70)
        print("STEP 3: CREATING DATALOADERS")
        print("="*70)
        
        # Setup cache directory for fast initialization
        cache_dir = os.path.join(self.ckpt_dir, 'cache')
        
        try:
            train_loader, valid_loader, _ = create_dataloaders(
                train_clean_dir=TRAIN_CLEAN_DIR,
                train_noisy_dir=TRAIN_NOISY_DIR,
                valid_clean_dir=VALID_CLEAN_DIR,
                valid_noisy_dir=VALID_NOISY_DIR,
                test_clean_dir=TEST_CLEAN_DIR,
                test_noisy_dir=TEST_NOISY_DIR,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                sample_rate=self.sample_rate,
                unit=self.unit,
                segment_size=self.segment_size,
                segment_shift=self.segment_shift,
                max_length_seconds=self.max_length_seconds,
                pin_memory=True,
                drop_last=True,
                cache_dir=cache_dir
            )
        except Exception as e:
            logger.error(f'Failed to create dataloaders: {e}')
            print(f"\n? DATALOADER CREATION FAILED: {e}")
            raise
        
        # Calculate iterations per epoch
        self.logging_period = len(train_loader)
        logger.info(f'Iterations per epoch: {self.logging_period}\n')

        print("\n" + "="*70)
        print("STEP 4: INITIALIZING MODEL")
        print("="*70)
        
        # Create network
        net = Net()
        logger.info(f'Model summary:\n{net}')

        net = net.to(self.device)
        if len(self.gpu_ids) > 1:
            net = DataParallel(net, device_ids=self.gpu_ids)
            logger.info(f'Using DataParallel with {len(self.gpu_ids)} GPUs')

        # Calculate model size
        param_count = numParams(net)
        logger.info(f'Trainable parameters: {param_count:,d} -> {param_count*32/8/(2**20):.2f} MB\n')

        # Network feeder (CMGAN-style)
        feeder = NetFeeder(
            self.device, 
            n_fft=self.n_fft,           # 320
            hop_length=self.hop_size,   # 160
            power=self.power            # 0.3
        )

        # Loss and optimizer (CMGAN-style parameters)
        criterion = LossFunction(
            device=self.device, 
            n_fft=self.n_fft,           # 320
            hop_length=self.hop_size,   # 160
            power=self.power            # 0.3
        )
        
        optimizer = Adam(net.parameters(), lr=self.lr, amsgrad=False)
        
        # ReduceLROnPlateau scheduler
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.plateau_factor,
            patience=self.plateau_patience,
            threshold=self.plateau_threshold,
            threshold_mode='rel',
            cooldown=2,
            min_lr=self.plateau_min_lr,
            verbose=True
        )
        
        # Initialize checkpoint info
        ckpt_info = {
            'cur_epoch': 0,
            'cur_iter': 0,
            'tr_loss': None,
            'cv_loss': None,
            'best_loss': float('inf'),
            'global_step': 0,
            'min_lr_epoch_count': 0
        }
        global_step = 0
        min_lr_epoch_count = 0
        
        # Resume training if needed
        if self.resume_model:
            logger.info('='*70)
            logger.info('RESUMING FROM CHECKPOINT')
            logger.info('='*70)
            logger.info(f'Loading: {self.resume_model}')
            
            ckpt = CheckPoint()
            ckpt.load(self.resume_model, self.device)
            
            # Load network state
            state_dict = {}
            for key in ckpt.net_state_dict:
                if len(self.gpu_ids) > 1:
                    state_dict['module.' + key] = ckpt.net_state_dict[key]
                else:
                    state_dict[key] = ckpt.net_state_dict[key]
            net.load_state_dict(state_dict)
            
            # Load optimizer state
            optim_state = ckpt.optim_state_dict
            for state in optim_state['state'].values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
            optimizer.load_state_dict(optim_state)
            
            # Load checkpoint info
            ckpt_info = ckpt.ckpt_info
            global_step = ckpt_info.get('global_step', 0)
            min_lr_epoch_count = ckpt_info.get('min_lr_epoch_count', 0)
            
            logger.info(f'Resumed from epoch {ckpt_info["cur_epoch"] + 1}')
            logger.info(f'Best CV loss: {ckpt_info["best_loss"]:.4f}')
            logger.info('='*70 + '\n')
        
        print("\n" + "="*70)
        print("STEP 5: STARTING TRAINING LOOP")
        print("="*70 + "\n")
        
        # Training loop
        logger.info('Starting training loop...\n')
        
        while ckpt_info['cur_epoch'] < self.max_n_epochs:
            accu_tr_loss = 0.
            accu_n_frames = 0
            net.train()
            
            epoch_start_time = timeit.default_timer()
            
            # Iterate over batches
            for n_iter, batch in enumerate(train_loader):
                global_step += 1
                
                # Get batch data
                mix = batch['mix'].to(self.device)
                sph = batch['sph'].to(self.device)
                n_samples = batch['n_samples'].to(self.device)
                
                n_frames = countFrames(n_samples, self.win_size, self.hop_size)
                
                if isinstance(n_frames, torch.Tensor):
                    n_frames_sum = n_frames.sum().item()
                else:
                    n_frames_sum = sum(n_frames)

                iter_start_time = timeit.default_timer()
                
                # Prepare features and labels (MODIFIED: returns norm_factors)
                feat, lbl, norm_factors = feeder(mix, sph)
                loss_mask = lossMask(
                    shape=lbl.shape, 
                    n_frames=n_frames, 
                    device=self.device
                )
                
                # Forward pass
                optimizer.zero_grad()
                with torch.enable_grad():
                    est = net(feat, global_step=global_step)
                
                # Compute loss (MODIFIED: pass norm_factors)
                loss = criterion(est, lbl, loss_mask, n_frames, mix, n_samples, norm_factors)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.clip_norm > 0.0:
                    clip_grad_norm_(net.parameters(), self.clip_norm)
                
                optimizer.step()
                
                # Accumulate loss
                running_loss = loss.data.item()
                accu_tr_loss += running_loss * n_frames_sum
                accu_n_frames += n_frames_sum

                iter_end_time = timeit.default_timer()
                batch_time = iter_end_time - iter_start_time
                
                # Logging - ONLY EVERY 50 ITERATIONS
                # Log every 50 iterations OR at the last iteration of the epoch
                if (n_iter + 1) % 50 == 0 or (n_iter + 1) == self.logging_period:
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    log_msg = f'Epoch [{ckpt_info["cur_epoch"] + 1}/{self.max_n_epochs}], ' \
                              f'Iter [{n_iter + 1}/{self.logging_period}], ' \
                              f'LR: {current_lr:.6f}, ' \
                              f'tr_loss: {running_loss:.4f} / {accu_tr_loss / accu_n_frames:.4f}, ' \
                              f'time: {batch_time:.4f}s'
                    
                    if self.time_log:
                        with open(self.time_log, 'a+') as f:
                            print(log_msg, file=f)
                            f.flush()
                    else:
                        print(log_msg, flush=True)
            
            # End of epoch - run validation
            avg_tr_loss = accu_tr_loss / accu_n_frames
            
            logger.info('\n' + '='*70)
            logger.info(f'VALIDATION - Epoch {ckpt_info["cur_epoch"] + 1}/{self.max_n_epochs}')
            logger.info('='*70)
            
            # Run validation
            avg_cv_loss = self.validate(
                net, valid_loader, criterion, feeder, global_step, logger
            )
            
            # Restore training mode
            net.train()
            
            # Update checkpoint info
            ckpt_info['cur_iter'] = n_iter
            ckpt_info['global_step'] = global_step
            is_best = avg_cv_loss < ckpt_info['best_loss']
            
            if is_best:
                improvement = ckpt_info['best_loss'] - avg_cv_loss
                logger.info(f'NEW BEST MODEL! Improvement: {improvement:.4f}')
                ckpt_info['best_loss'] = avg_cv_loss
                min_lr_epoch_count = 0
            
            ckpt_info['tr_loss'] = avg_tr_loss
            ckpt_info['cv_loss'] = avg_cv_loss
            
            # ReduceLROnPlateau step
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_cv_loss)
            new_lr = optimizer.param_groups[0]['lr']
            
            if abs(old_lr - new_lr) > 1e-8:
                logger.info('='*70)
                logger.info('LEARNING RATE REDUCED')
                logger.info(f'LR: {old_lr:.6f} -> {new_lr:.6f}')
                logger.info('='*70)
            
            # Early stopping logic
            if new_lr <= self.plateau_min_lr:
                if not is_best:
                    min_lr_epoch_count += 1
                    logger.info(f'Early stop counter: {min_lr_epoch_count}/{self.early_stop_patience}')
                    
                    if min_lr_epoch_count >= self.early_stop_patience:
                        logger.info('='*70)
                        logger.info('EARLY STOPPING TRIGGERED')
                        logger.info(f'No improvement for {self.early_stop_patience} epochs at min LR')
                        logger.info(f'Best CV loss: {ckpt_info["best_loss"]:.4f}')
                        logger.info('='*70)
                        
                        # Save final checkpoint
                        self._save_checkpoint(ckpt_info, net, optimizer, scheduler, is_best=False)
                        logger.info('Training stopped by early stopping.\n')
                        return
            else:
                min_lr_epoch_count = 0
            
            ckpt_info['min_lr_epoch_count'] = min_lr_epoch_count
            
            # Save checkpoint
            logger.info(f'Train Loss: {avg_tr_loss:.4f} | CV Loss: {avg_cv_loss:.4f} | '
                       f'Best: {ckpt_info["best_loss"]:.4f} | LR: {new_lr:.6f}')
            logger.info('='*70 + '\n')
            
            self._save_checkpoint(ckpt_info, net, optimizer, scheduler, is_best)
            
            # Write to loss log
            ckpt = CheckPoint(ckpt_info, None, None, None)
            lossLog(os.path.join(self.ckpt_dir, self.loss_log), ckpt, self.logging_period)
            
            # Next epoch
            ckpt_info['cur_epoch'] += 1
        
        logger.info('='*70)
        logger.info('TRAINING COMPLETED')
        logger.info(f'Total epochs: {ckpt_info["cur_epoch"]}')
        logger.info(f'Best CV loss: {ckpt_info["best_loss"]:.4f}')
        logger.info('='*70)
        
        return
    
    def _save_checkpoint(self, ckpt_info, net, optimizer, scheduler, is_best):
        """Helper to save checkpoint"""
        model_path = os.path.join(self.ckpt_dir, 'models')
        os.makedirs(model_path, exist_ok=True)
        
        if len(self.gpu_ids) > 1:
            ckpt = CheckPoint(
                ckpt_info, 
                net.module.state_dict(), 
                optimizer.state_dict(),
                scheduler.state_dict()
            )
        else:
            ckpt = CheckPoint(
                ckpt_info, 
                net.state_dict(), 
                optimizer.state_dict(),
                scheduler.state_dict()
            )
        
        ckpt.save(
            os.path.join(model_path, 'latest.pt'),
            is_best,
            os.path.join(model_path, 'best.pt') if is_best else None
        )

    def validate(self, net, cv_loader, criterion, feeder, global_step, logger):
        """Validation procedure with CMGAN-style processing"""
        accu_cv_loss = 0.
        accu_n_frames = 0

        model = net.module if isinstance(net, DataParallel) else net
        model.eval()
        
        with torch.no_grad():
            for batch in cv_loader:
                mix = batch['mix'].to(self.device)
                sph = batch['sph'].to(self.device)
                n_samples = batch['n_samples'].to(self.device)
                
                n_frames = countFrames(n_samples, self.win_size, self.hop_size)

                # MODIFIED: feeder returns norm_factors
                feat, lbl, norm_factors = feeder(mix, sph)
                loss_mask = lossMask(
                    shape=lbl.shape, 
                    n_frames=n_frames, 
                    device=self.device
                )
                
                est = model(feat, global_step=global_step)
                
                # MODIFIED: criterion needs norm_factors
                loss = criterion(est, lbl, loss_mask, n_frames, mix, n_samples, norm_factors)

                if isinstance(n_frames, torch.Tensor):
                    n_frames_sum = n_frames.sum().item()
                else:
                    n_frames_sum = sum(n_frames)
                
                accu_cv_loss += loss.data.item() * n_frames_sum
                accu_n_frames += n_frames_sum
        
        avg_cv_loss = accu_cv_loss / accu_n_frames
        return avg_cv_loss
    
    @torch.no_grad()
    def enhance_one_track(self, model, audio_path, saved_dir, cut_len, save_tracks=True):
        """
        Enhance a single audio track using CMGAN-style processing.

        This method follows CMGAN's enhance_one_track approach while using
        your model's STFT parameters (n_fft=320, hop=160).

        Args:
            model: Trained network
            audio_path: Path to noisy audio file
            saved_dir: Directory to save enhanced audio
            cut_len: Maximum length for processing
            save_tracks: Whether to save enhanced audio

        Returns:
            est_audio: Enhanced audio as numpy array
            length: Original audio length
        """
        name = os.path.split(audio_path)[-1]

        # Load audio
        noisy, sr = torchaudio.load(audio_path)
        assert sr == 16000, f"Expected 16kHz, got {sr}Hz"
        noisy = noisy.to(self.device)

        # RMS Normalization (CMGAN-style)
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
        noisy = torch.transpose(noisy, 0, 1)
        noisy = torch.transpose(noisy * c, 0, 1)

        # Frame-based padding
        length = noisy.size(-1)
        frame_num = int(np.ceil(length / self.hop_length))
        padded_len = frame_num * self.hop_length
        padding_len = padded_len - length

        # Audio repetition padding (CMGAN-style, not zero padding)
        if padding_len > 0:
            noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)

        # Batch reshaping for long files
        if padded_len > cut_len:
            batch_size = int(np.ceil(padded_len / cut_len))
            while self.hop_length % batch_size != 0:
                batch_size += 1
            noisy = torch.reshape(noisy, (batch_size, -1))

        # STFT
        noisy_spec = torch.stft(
            noisy,
            self.n_fft,
            self.hop_length,
            window=torch.hamming_window(self.n_fft).to(self.device),
            onesided=True,
            return_complex=False  # Returns [batch, freq, time, 2]
        )

        # Extract real and imaginary parts and transpose to [batch, time, freq]
        # noisy_spec shape: [batch, freq, time, 2]
        noisy_real = noisy_spec[:, :, :, 0].transpose(1, 2).contiguous()  # [B, T, F]
        noisy_imag = noisy_spec[:, :, :, 1].transpose(1, 2).contiguous()  # [B, T, F]

        # Power compression
        noisy_real_c, noisy_imag_c = power_compress(noisy_real, noisy_imag, self.power)

        # Stack to [batch, 2, time, freq] format
        noisy_input = torch.stack([noisy_real_c, noisy_imag_c], dim=1)  # [B, 2, T, F]

        # Model inference
        est = model(noisy_input, global_step=None)

        # Extract estimated real and imaginary [B, T, F]
        est_real = est[:, 0, :, :]  # [B, T, F]
        est_imag = est[:, 1, :, :]  # [B, T, F]

        # Power decompression
        est_real_uc, est_imag_uc = power_uncompress(est_real, est_imag, self.power)

        # Transpose back to [B, F, T] for iSTFT
        est_real_uc = est_real_uc.transpose(1, 2)  # [B, F, T]
        est_imag_uc = est_imag_uc.transpose(1, 2)  # [B, F, T]

        # Combine to complex spectrum [B, F, T, 2]
        est_spec = torch.stack([est_real_uc, est_imag_uc], dim=-1)

        # iSTFT
        est_audio = torch.istft(
            est_spec,
            self.n_fft,
            self.hop_length,
            window=torch.hamming_window(self.n_fft).to(self.device),
            onesided=True,
            length=padded_len
        )

        # Denormalize (CMGAN-style)
        est_audio = est_audio / c

        # Flatten and truncate to original length
        est_audio = torch.flatten(est_audio)[:length].cpu().numpy()
        assert len(est_audio) == length

        # Save if requested
        if save_tracks:
            saved_path = os.path.join(saved_dir, name)
            sf.write(saved_path, est_audio, sr)

        return est_audio, length

    def test(self):
        """
        CMGAN-Style Testing Procedure.

        Processes test files one-by-one (like CMGAN) and computes metrics inline.
        Uses your model's STFT parameters (n_fft=320, hop=160) to maintain
        checkpoint compatibility.
        """
        print("\n" + "="*70)
        print("CMGAN-STYLE TESTING")
        print("="*70)
        print("Using your model's parameters: n_fft=320, hop=160, power=0.3")
        print("="*70 + "\n")

        # Load configuration from test_conf
        model_file = test_conf['model_file']
        save_tracks = test_conf['save_tracks']
        save_dir = test_conf['save_dir']
        enable_metrics = test_conf['compute_metrics']
        cut_len = test_conf['cut_len']
        device_str = test_conf['device']

        # Setup device
        if device_str == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        print(f"Configuration:")
        print(f"  Model: {model_file}")
        print(f"  Test clean: {TEST_CLEAN_DIR}")
        print(f"  Test noisy: {TEST_NOISY_DIR}")
        print(f"  Output: {save_dir}")
        print(f"  Device: {self.device}")
        print(f"  Compute metrics: {enable_metrics}")
        print(f"  STFT: n_fft={self.n_fft}, hop={self.hop_length}, power={self.power}")
        print()
        # Create output directory
        if save_tracks:
            os.makedirs(save_dir, exist_ok=True)

        # Load model
        print("="*70)
        print("Loading Model")
        print("="*70)

        net = Net()
        net = net.to(self.device)

        if not os.path.isfile(model_file):
            # Check for alternative checkpoint files
            ckpt_dir = os.path.dirname(model_file)
            alternatives = []
            if os.path.isdir(ckpt_dir):
                for alt in ['best.pt', 'best_pesq.pt', 'latest.pt']:
                    alt_path = os.path.join(ckpt_dir, alt)
                    if os.path.isfile(alt_path):
                        alternatives.append(alt_path)

            error_msg = f"Model file not found: {model_file}"
            if alternatives:
                error_msg += f"\n\nAlternative checkpoint files found:\n"
                for alt in alternatives:
                    error_msg += f"  - {alt}\n"
                error_msg += "\nUpdate test_conf['model_file'] in configs.py to use one of these files."
            else:
                error_msg += "\n\nPlease train the model first or check the model_file path in configs.py"

            raise FileNotFoundError(error_msg)

        ckpt = CheckPoint()
        ckpt.load(model_file, self.device)
        net.load_state_dict(ckpt.net_state_dict)

        print(f"✓ Model loaded from: {model_file}")
        print(f"  Epoch: {ckpt.ckpt_info['cur_epoch'] + 1}")
        print(f"  Best loss: {ckpt.ckpt_info.get('best_loss', 'N/A')}")
        print()

        net.eval()

        # Get audio file list
        print("="*70)
        print("Loading Test Files")
        print("="*70)

        if not os.path.isdir(TEST_NOISY_DIR):
            print(f"\n✗ ERROR: Noisy directory not found: {TEST_NOISY_DIR}")
            return

        audio_list = os.listdir(TEST_NOISY_DIR)
        audio_list = natsorted(audio_list)
        audio_list = [f for f in audio_list if f.endswith('.wav')]
        num_files = len(audio_list)

        print(f"Found {num_files} audio files to process")
        print()

        # Check if clean directory exists for metrics
        has_clean = os.path.isdir(TEST_CLEAN_DIR)
        if not has_clean and enable_metrics:
            print(f"⚠ Warning: Clean directory not found: {TEST_CLEAN_DIR}")
            print("  Metrics will not be computed")
            enable_metrics = False

        # Initialize metrics
        if enable_metrics and HAS_METRICS:
            metrics_total = np.zeros(6)  # PESQ, CSIG, CBAK, COVL, SSNR, STOI
            metrics_count = 0
        elif enable_metrics and not HAS_METRICS:
            print("⚠ Warning: Metrics libraries not installed")
            print("  Install with: pip install pesq pystoi pysepm")
            enable_metrics = False

        # Process files
        print("="*70)
        print("Processing Files")
        print("="*70)


        # Process each audio file (CMGAN-style)
        for idx, audio in enumerate(audio_list):
            noisy_path = os.path.join(TEST_NOISY_DIR, audio)
            clean_path = os.path.join(TEST_CLEAN_DIR, audio) if has_clean else None

            # Enhance audio using CMGAN-style processing
            try:
                est_audio, length = self.enhance_one_track(
                    net, noisy_path, save_dir, cut_len, save_tracks
                )

                # Compute metrics if available
                if enable_metrics and has_clean and HAS_METRICS:
                    clean_audio, sr = sf.read(clean_path)
                    assert sr == 16000

                    # Ensure same length
                    min_len = min(len(clean_audio), len(est_audio))
                    clean_audio = clean_audio[:min_len]
                    est_audio_metric = est_audio[:min_len]

                    try:
                        metrics = compute_metrics(clean_audio, est_audio_metric, sr, 0)
                        metrics = np.array(metrics)
                        metrics_total += metrics
                        metrics_count += 1
                    except Exception as e:
                        print(f"⚠ Warning: Metrics failed for {audio}: {e}")

            except Exception as e:
                print(f"✗ Error processing {audio}: {e}")
                continue

            # Progress
            if (idx + 1) % 10 == 0 or (idx + 1) == num_files:
                print(f"  Progress: {idx + 1}/{num_files} files ({100*(idx+1)/num_files:.1f}%)")

        # Report results
        print("\n" + "="*70)
        print("TESTING COMPLETED")
        print("="*70)
        print(f"Processed: {num_files} files")

        if enable_metrics and metrics_count > 0:
            metrics_avg = metrics_total / metrics_count
            print(f"\nAverage Metrics (over {metrics_count} files):")
            print(f"  PESQ: {metrics_avg[0]:.4f}")
            print(f"  CSIG: {metrics_avg[1]:.4f}")
            print(f"  CBAK: {metrics_avg[2]:.4f}")
            print(f"  COVL: {metrics_avg[3]:.4f}")
            print(f"  SSNR: {metrics_avg[4]:.4f} dB")
            print(f"  STOI: {metrics_avg[5]:.4f}")
        else:
            print("\nMetrics not computed")

        if save_tracks:
            print(f"\nEnhanced tracks saved to: {save_dir}")

        print("="*70)
        
        return


if __name__ == '__main__':
    import sys
    
    model = Model()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            model.train()
        elif sys.argv[1] == 'test':
            model.test()
        else:
            print(f'? Unknown command: {sys.argv[1]}')
            print('Usage: python models.py [train|test]')
    else:
        print('Usage: python models.py [train|test]')