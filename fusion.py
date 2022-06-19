import os
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

    def saveAndDisplay(img, path, title='Image'):
        ImageUtils.save(img, path)
        savedImg = ImageUtils.read(path)
        ImageUtils.display(savedImg, title)


class RwtFilter:
    den = 4 * (2**0.5)
    root3 = 3**0.5
    c0 = (1 + root3) / den
    c1 = (3 + root3) / den
    c2 = (3 - root3) / den
    c3 = (1 - root3) / den

    # 1D DWT low pass filter
    h = np.array([[ c0, c1, c2, c3]])

    # 1D DWT high pass filter
    g = np.array([[ c3, -c2, c1, -c0]])

    # === 2d RWT Filters ===
    # No Edge
    H_AA = np.matmul(np.transpose(h), h)

    # -45
    H_AD = np.matmul(np.transpose(h), g)

    # +45
    H_DA = np.matmul(np.transpose(g), h)

    # 0 and 90
    H_DD = np.matmul(np.transpose(g), g)

    list = [H_AA, H_AD, H_DA, H_DD]

    dict = {
        'AA': H_AA,
        'AD': H_AD,
        'DA': H_DA,
        'DD': H_DD,
    }

class Fusion:

    def __init__(self, input_path, output_path) -> None:
        
        ct = ImageUtils.read(os.path.join(input_path, 'ct.jpg'))
        mri = ImageUtils.read(os.path.join(input_path, 'mri.jpg'))

        self.images = {
            'CT': ct,
            'MRI': mri
        }

        self.rwtImages = {}

        self.fusedImages = {}

        self.output_path = output_path

    def max(img1, img2):
        res = np.zeros((len(img1), len(img1[0])), dtype=np.uint8)
        for i in range(len(img1)):
            for j in range(len(img1[0])):
                res[i][j] = max([img1[i][j], img2[i][j]])
        return res

    def entropy1d(signal):
        lensig = signal.size
        symset = list(set(signal))
        propab = [np.size(signal[signal==i])/(1.0*lensig) for i in symset]
        ent = np.sum([p*np.log2(1.0/p) for p in propab])
        return ent

    def entropy2d(image):
        N = 10
        S = image.shape
        E = np.array(image)
        for row in range(S[0]):
            for col in range(S[1]):
                Lx = np.max([0,col-N])
                Ux = np.min([S[1],col+N])
                Ly = np.max([0,row-N])
                Uy = np.min([S[0],row+N])
                region = image[Ly:Uy,Lx:Ux].flatten()
                E[row,col] = Fusion.entropy1d(region)
        return E

    def entropy_max(ct_k, mri_k, en_ct_k, en_mri_k):
        en_max = np.zeros((256, 256))
        for i in range(256):
            for j in range(256):
                en_max[i][j] = ct_k[i][j] if en_ct_k[i][j] >= en_mri_k[i][j] else mri_k[i][j]
        return en_max
    
    def fuse(self):

        for imgName, image in zip(self.images, self.images.values()):
            for filterName, filter in zip(RwtFilter.dict, RwtFilter.dict.values()):
                res = cv2.filter2D(image, -1, filter)
                name = f'{imgName}_{filterName}'
                self.rwtImages[name] = res
                ImageUtils.saveAndDisplay(res, os.path.join(self.output_path, f'{name}.jpg'), name)
        
        for i in ['CT', 'MRI']:
            for f in ['AD', 'DA', 'DD']:
                cur = f'{i}_{f}'
                img = self.rwtImages[cur]
                en = Fusion.entropy2d(img)
                self.rwtImages[f'EN_{cur}'] = en
                ImageUtils.save(en, os.path.join(self.output_path, f'EN_{cur}.jpg'))

        fused_aa = Fusion.max(self.rwtImages['CT_AA'], self.rwtImages['MRI_AA'])
        ImageUtils.saveAndDisplay(fused_aa, os.path.join(self.output_path, 'FUSED_AA.jpg'), 'FUSED_AA')
        self.fusedImages['FUSED_AA'] = fused_aa        

        for k in ['AD', 'DA', 'DD']:
            fused_k = Fusion.entropy_max(self.rwtImages[f'CT_{k}'], self.rwtImages[f'MRI_{k}'], self.rwtImages[f'EN_CT_{k}'], self.rwtImages[f'EN_MRI_{k}'])
            name = f'FUSED_{k}'
            self.fusedImages[name] = fused_k
            ImageUtils.saveAndDisplay(fused_k, os.path.join(self.output_path, f'{name}.jpg'), name)

        coeffs = self.fusedImages['FUSED_AA'], (self.fusedImages['FUSED_AD'], self.fusedImages['FUSED_DA'], self.fusedImages['FUSED_DD'])
        fused_img = pywt.idwt2(coeffs, 'db4')
        ImageUtils.saveAndDisplay(fused_img, os.path.join(self.output_path, 'FUSED.jpg'), 'Fused Image')