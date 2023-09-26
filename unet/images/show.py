import os
import glob
import imageio
import matplotlib.pyplot as plt

imgs = glob.glob("original/*.jpg")
index = 0

path1 = imgs[index]
path2 = os.path.join("masks", os.path.basename(path1).replace(".jpg", ".png"))


img_view  = imageio.imread(path1)
mask_view = imageio.imread(path2)
print(os.path.basename(path1), img_view.shape, mask_view.shape)

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(15, 15))
ax1.imshow(img_view)
ax1.set_title(f"Image #{index}")
ax2.imshow(mask_view)
ax2.set_title("Masked Image")
plt.show()
