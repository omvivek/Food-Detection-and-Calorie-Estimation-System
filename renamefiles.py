import os


image_dir = 'dataset/images/val/rice/'  
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg','.png'))]
for i, image in enumerate(image_files):
    new_name = f'rice_{i+1}.jpg'  
    os.rename(os.path.join(image_dir, image), os.path.join(image_dir, new_name))
    #print(f"Renamed {image} to {new_name}")
