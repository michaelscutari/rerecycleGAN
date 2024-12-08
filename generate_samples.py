# generate_samples.py
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from model import RecycleGAN
import cv2
import numpy as np

class StyleTransfer:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((240, 432)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Load model
        self.model = RecycleGAN(
            l_adv=1.0, l_cycle=10.0, l_iden=2.0, l_temp=2.0,
            learning_rate_d=0.0002, learning_rate_g=0.0002, learning_rate_p=0.0002
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.model.to(device)

    def preprocess_image(self, image_path):
        """Process a single image for inference"""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image.unsqueeze(0)

    def denormalize(self, tensor):
        """Convert tensor to image"""
        tensor = tensor.clone().detach()
        tensor = tensor * 0.5 + 0.5
        tensor = tensor.clamp(0, 1)
        tensor = tensor * 255
        tensor = tensor.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        return tensor

    def transfer_style(self, input_path, output_path, direction='AtoB'):
        """
        Transfer style of input image
        Args:
            input_path: Path to input image
            output_path: Path to save output image
            direction: 'AtoB' or 'BtoA'
        """
        # Prepare input
        image = self.preprocess_image(input_path).to(self.device)
        
        with torch.no_grad():
            if direction == 'AtoB':
                output = self.model.AtoB(image)
            else:
                output = self.model.BtoA(image)
        
        # Convert to image and save
        output_image = self.denormalize(output[0])
        cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
        
    def process_video(self, input_video, output_video, direction='AtoB'):
        """
        Process entire video
        Args:
            input_video: Path to input video
            output_video: Path to save output video
            direction: 'AtoB' or 'BtoA'
        """
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (432, 240))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame
            frame = cv2.resize(frame, (432, 240))
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Transform
            frame_tensor = self.transform(frame_pil).unsqueeze(0).to(self.device)
            
            # Generate
            with torch.no_grad():
                if direction == 'AtoB':
                    output = self.model.AtoB(frame_tensor)
                else:
                    output = self.model.BtoA(frame_tensor)
            
            # Convert to image
            output_frame = self.denormalize(output[0])
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            
            # Write frame
            out.write(output_frame)
            
        cap.release()
        out.release()

# Usage example
if __name__ == "__main__":
    # Initialize
    transfer = StyleTransfer(
        checkpoint_path="./checkpoints/unk/recyclegan-epochepoch=9-v3.ckpt",
        device='cuda'
    )
    
    # Process single image
    transfer.transfer_style(
        input_path="./inference/extracted_frames/frame_0017.png",
        output_path="./inference/output33.png",
        direction='AtoB'  # or 'BtoA'
    )
    
    # Process video
    # transfer.process_video(
    #     input_video="input.mp4",
    #     output_video="output.mp4",
    #     direction='AtoB'  # or 'BtoA'
    # )