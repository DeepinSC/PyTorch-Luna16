import cv2
import SimpleITK as sitk

filename = "data/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd"
ds = sitk.ReadImage(filename)
img_array = sitk.GetArrayFromImage(ds)
frame_num, width, height = img_array.shape