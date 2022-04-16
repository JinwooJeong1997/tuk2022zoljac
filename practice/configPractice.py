import rawpy
import imageio

with rawpy.imread('90.ARW') as raw:
    rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)

# Extract Red, Green and Blue channels and save as separate files
R = rgb[:,:,0]
G = rgb[:,:,1]
B = rgb[:,:,2]

imageio.imsave('R.tif', R)
imageio.imsave('G.tif', G)
imageio.imsave('B.tif', B)