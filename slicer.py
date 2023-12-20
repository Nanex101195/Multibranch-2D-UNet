#########################################
#       niftii2png for Python 3.8       #
#         NIfTI Image Converter         #
#                v1.1                   #
#                                       #
#       Written by Florian Raab         #
#  Florian.Raab@stud.uni-regensburg.de  #
#              19 Dec 2020              #
#########################################
#
#
# Fetches all files from directory of trainings images and labels.
# Distinguishes between different modalities and creates new directories for
# every of them (probably handy for different branches?)
# Creates 2d slices of axial, coronal and sagittal views and saves them to disk.

# image array[:,:,x] is axial
# imageArray[:,x,:] is coronal
# imageArray[x,:,:] is sagittal


import scipy
import numpy
import os
import nibabel as nb
from glob import glob
import skimage.transform as skTrans
import cv2

sub_identifier = ["training_01", "training_02", "training_03", "training_04", "training_05"]
msk_identifier = ["mask1", "mask2"]

# Substrings to distinguish between different modalities in filename
substr_flair = 'FLAIR'
substr_t1 = 'T1'
substr_t2 = 'T2'
substr_pd = 'PD'

# Boolean variables to specify, which perspectives of volume should be sliced
SLICE_AXIAL = True
SLICE_CORONAL = True
SLICE_SAGITTAL = True

# 512x512x512 results in max count of a 3 digit number per dimension
# --> 001 as well as 510 for example
SLICE_DECIMATE_IDENTIFIER = 3

# function for normalizing image intensity of volumes to values between 0 and 1 (float)
# MAX and MIN is different for every single volume, which is taken care of


def norm_img_int_rng(img, MIN_INTENSITY, MAX_INTENSITY):
	INTENSITY_RANGE = MAX_INTENSITY - MIN_INTENSITY
	img[img > MAX_INTENSITY] = MAX_INTENSITY
	img[img < MIN_INTENSITY] = MIN_INTENSITY
	return (img - MIN_INTENSITY) / INTENSITY_RANGE


# function for reading in .nii files
def read_img_vol(imgPath, normalize=False):
	a = nb.load(imgPath).get_fdata()
	img = a
	if normalize:
		return norm_img_int_rng(img, numpy.min(img), numpy.max(img))
	else:
		return img

# function for saving one slice of the volume, after converting the float64 to an
# uint8.
# By normalizing the float64s and then multiplying it with maximum dynamic range of
# PNG before converting to an uint8, we can maintain the best Intensity Range and
# get rid of clipping by conversion
def save_slice(IMG_SIZE, img, fname, path):
	IMAGE_HEIGHT, IMAGE_WIDTH = IMG_SIZE
	img = numpy.uint8(img*255)
	fout = os.path.join(path, f'{fname}.png')
	img = cv2.resize(img, dsize=(IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
	cv2.imwrite(fout, img)
	print(f'[+] Slice saved: {fout}', end='\r')


def find_and_load_data(modalities, mask_id, subj_base, subj):
	subj_timepoint = subj_base + "/" + str(subj)
	flair_base = subj_timepoint + modalities[0] + ".nii"
	t1_base = subj_timepoint + modalities[1] + ".nii"
	t2_base = subj_timepoint + modalities[2] + ".nii"
	pd_base = subj_timepoint + modalities[3] + ".nii"
	msk1_base = subj_timepoint + mask_id[0] + ".nii"
	msk2_base = subj_timepoint + mask_id[1] + ".nii"

	flair_img = read_img_vol(flair_base, True)
	t1_img = read_img_vol(t1_base, True)
	t2_img = read_img_vol(t2_base, True)
	pd_img = read_img_vol(pd_base, True)
	msk1_img = read_img_vol(msk1_base, False)
	msk2_img = read_img_vol(msk2_base, False)

	return flair_img, t1_img, t2_img, pd_img, msk1_img, msk2_img


# function for saving ALL slices of the volume to disk.
# perspectives can be chosen at the beginning of the script
def slice_and_save_vol_img(IMG_SIZE, vol_lbl, vol_flair, vol_t1, vol_t2, vol_pd, fname, path_flair, path_t1, path_t2,
						   path_pd, path_lbl, train=True):
	(dimx, dimy, dimz) = vol_lbl.shape
	cnt = 0
	cnt_sag = 0
	cnt_cor = 0
	cnt_axi = 0

	if SLICE_SAGITTAL:
		print('Slicing sagittal: ')
		for i in range(dimx):
			if train == True:
				if numpy.all(vol_lbl[i, :, :] == 0):
					cnt_sag += 1
					if (cnt_sag%30 == 0):
						cnt+=1
						save_slice(IMG_SIZE, vol_lbl[i,:,:],
								   fname+f'sagittal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_lbl)
						save_slice(IMG_SIZE, vol_flair[i,:,:],
								   fname+f'sagittal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_flair)
						save_slice(IMG_SIZE, vol_t1[i,:,:],
								   fname+f'sagittal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_t1)
						save_slice(IMG_SIZE, vol_t2[i,:,:],
								   fname+f'sagittal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_t2)
						save_slice(IMG_SIZE, vol_pd[i,:,:],
								   fname+f'sagittal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_pd)
				else:
					cnt+=1
					save_slice(IMG_SIZE, vol_lbl[i,:,:],
							   fname+f'sagittal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_lbl)
					save_slice(IMG_SIZE, vol_flair[i,:,:],
							   fname+f'sagittal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_flair)
					save_slice(IMG_SIZE, vol_t1[i,:,:],
							   fname+f'sagittal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_t1)
					save_slice(IMG_SIZE, vol_t2[i,:,:],
							   fname+f'sagittal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_t2)
					save_slice(IMG_SIZE, vol_pd[i,:,:],
							   fname+f'sagittal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_pd)
			else:
				cnt+=1
				save_slice(IMG_SIZE, vol_lbl[i,:,:],
						   fname+f'sagittal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_lbl)
				save_slice(IMG_SIZE, vol_flair[i,:,:],
						   fname+f'sagittal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_flair)
				save_slice(IMG_SIZE, vol_t1[i,:,:],
						   fname+f'sagittal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_t1)
				save_slice(IMG_SIZE, vol_t2[i,:,:],
						   fname+f'sagittal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_t2)
				save_slice(IMG_SIZE, vol_pd[i,:,:],
						   fname+f'sagittal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_pd)


	if SLICE_CORONAL:
		print('Slicing coronal: ')
		for i in range(dimy):
			if train == True:
				if numpy.all(vol_lbl[:,i,:] == 0):
					cnt_cor += 1
					if (cnt_cor%30 == 0):
						cnt+=1
						save_slice(IMG_SIZE, vol_lbl[:,i,:],
								   fname+f'coronal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_lbl)
						save_slice(IMG_SIZE, vol_flair[:,i,:],
								   fname+f'coronal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_flair)
						save_slice(IMG_SIZE, vol_t1[:,i,:],
								   fname+f'coronal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_t1)
						save_slice(IMG_SIZE, vol_t2[:,i,:],
								   fname+f'coronal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_t2)
						save_slice(IMG_SIZE, vol_pd[:,i,:],
								   fname+f'coronal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_pd)
				else:
					cnt+=1
					save_slice(IMG_SIZE, vol_lbl[:,i,:],
							   fname+f'coronal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_lbl)
					save_slice(IMG_SIZE, vol_flair[:,i,:],
							   fname+f'coronal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_flair)
					save_slice(IMG_SIZE, vol_t1[:,i,:],
							   fname+f'coronal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_t1)
					save_slice(IMG_SIZE, vol_t2[:,i,:],
							   fname+f'coronal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_t2)
					save_slice(IMG_SIZE, vol_pd[:,i,:],
							   fname+f'coronal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_pd)
			else:
				cnt+=1
				save_slice(IMG_SIZE, vol_lbl[:,i,:],
						   fname+f'coronal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_lbl)
				save_slice(IMG_SIZE, vol_flair[:,i,:],
						   fname+f'coronal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_flair)
				save_slice(IMG_SIZE, vol_t1[:,i,:],
						   fname+f'coronal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_t1)
				save_slice(IMG_SIZE, vol_t2[:,i,:],
						   fname+f'coronal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_t2)
				save_slice(IMG_SIZE, vol_pd[:,i,:],
						   fname+f'coronal_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_pd)

	if SLICE_AXIAL:
		print('Slicing axial: ')
		for i in range(dimz):
			if train == True:
				if numpy.all(vol_lbl[:,:,i]==0):
					cnt_axi += 1
					if (cnt_axi%30 == 0):
						cnt+=1
						save_slice(IMG_SIZE, vol_lbl[:,:,i],
								   fname+f'axial_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_lbl)
						save_slice(IMG_SIZE, vol_flair[:,:,i],
								   fname+f'axial_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_flair)
						save_slice(IMG_SIZE, vol_t1[:,:,i],
								   fname+f'axial_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_t1)
						save_slice(IMG_SIZE, vol_t2[:,:,i],
								   fname+f'axial_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_t2)
						save_slice(IMG_SIZE, vol_pd[:,:,i],
								   fname+f'axial_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_pd)
				else:
					cnt+=1
					save_slice(IMG_SIZE, vol_lbl[:,:,i],
							   fname+f'axial_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_lbl)
					save_slice(IMG_SIZE, vol_flair[:,:,i],
							   fname+f'axial_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_flair)
					save_slice(IMG_SIZE, vol_t1[:,:,i],
							   fname+f'axial_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_t1)
					save_slice(IMG_SIZE, vol_t2[:,:,i],
							   fname+f'axial_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_t2)
					save_slice(IMG_SIZE, vol_pd[:,:,i],
							   fname+f'axial_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_pd)
			else:
				cnt+=1
				save_slice(IMG_SIZE, vol_lbl[:,:,i],
						   fname+f'axial_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_lbl)
				save_slice(IMG_SIZE, vol_flair[:,:,i],
						   fname+f'axial_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_flair)
				save_slice(IMG_SIZE, vol_t1[:,:,i],
						   fname+f'axial_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_t1)
				save_slice(IMG_SIZE, vol_t2[:,:,i],
						   fname+f'axial_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_t2)
				save_slice(IMG_SIZE, vol_pd[:,:,i],
						   fname+f'axial_slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}', path_pd)
	return cnt



if __name__ == "__main__":

	base_data = "~/Dokumente/Datasets/sample_dataset"
	
