import os
from statistics import mean
import numpy as np
import cv2
import pywt

class ImageUtils:
    def read(img_path):
        image = cv2.imread(img_path)
        # Convert to gray
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Resize to 256 x 256 px
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        return image

    def display(img, title='Image'):
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(img, path):
        cv2.imwrite(path, img)



class Fusion:

    def __init__(self, input_path, output_path) -> None:
        self.ct = ImageUtils.read(os.path.join(input_path, 'ct.jpg'))
        self.mri = ImageUtils.read(os.path.join(input_path, 'mri.jpg'))
        self.output_path = output_path
    
    def mean(img1, img2):
        res = np.zeros((len(img1), len(img1[0])), dtype=int)
        for i in range(len(img1)):
            for j in range(len(img1[0])):
                res[i][j] = mean([img1[i][j], img2[i][j]])
        return res

    def max(img1, img2):
        res = np.zeros((len(img1), len(img1[0])), dtype=int)
        for i in range(len(img1)):
            for j in range(len(img1[0])):
                res[i][j] = max([img1[i][j], img2[i][j]])
        return res

    def deconstruct(self, img):
        LL, (LH, HL, HH) = pywt.dwt2(img, 'haar')
        return LL, LH, HL, HH

    def reconstruct(self, LL, LH, HL, HH):
        coeffs = LL, (LH, HL, HH)
        fusion = pywt.idwt2(coeffs, 'haar')
        return fusion

    
    def fuse(self):

        LL_ct, LH_ct, HL_ct, HH_ct = self.deconstruct(self.ct)

        ImageUtils.display(LL_ct, 'CT H(AA)')
        ImageUtils.display(LH_ct, 'CT H(AD)')
        ImageUtils.display(HL_ct, 'CT H(DA)')
        ImageUtils.display(HH_ct, 'CT H(DD)')

        ImageUtils.save(LL_ct, os.path.join(self.output_path, 'ct-H-aa.jpg'))
        ImageUtils.save(LH_ct, os.path.join(self.output_path, 'ct-H-ad.jpg'))
        ImageUtils.save(HL_ct, os.path.join(self.output_path, 'ct-H-da.jpg'))
        ImageUtils.save(HH_ct, os.path.join(self.output_path, 'ct-H-dd.jpg'))

        LL_mri, LH_mri, HL_mri, HH_mri = self.deconstruct(self.mri)

        ImageUtils.display(LL_mri, 'MRI H(AA)')
        ImageUtils.display(LH_mri, 'MRI H(AD)')
        ImageUtils.display(HL_mri, 'MRI H(DA)')
        ImageUtils.display(HH_mri, 'MRI H(DD)')

        ImageUtils.save(LL_mri, os.path.join(self.output_path, 'mri-H-aa.jpg'))
        ImageUtils.save(LH_mri, os.path.join(self.output_path, 'mri-H-ad.jpg'))
        ImageUtils.save(HL_mri, os.path.join(self.output_path, 'mri-H-da.jpg'))
        ImageUtils.save(HH_mri, os.path.join(self.output_path, 'mri-H-dd.jpg'))

        LL = Fusion.mean(LL_ct, LL_mri)
        LH = Fusion.max(LH_ct, LH_mri)
        HL = Fusion.max(HL_ct, HL_mri)
        HH = Fusion.max(HH_ct, HH_mri)

        fused_img = self.reconstruct(LL, LH, HL, HH)

        ImageUtils.display(fused_img, 'Fused Image')
        ImageUtils.save(fused_img, os.path.join(self.output_path, 'fused.jpg'))