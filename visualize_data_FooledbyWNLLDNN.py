#!/usr/bin/python
"""
Visualize the images, original image and perturbed images
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys, os

#pkl_loc = sys.argv[1]
pkl_loc = "fooled_fgsd_WNLL.pkl"
with open(pkl_loc, "r") as f:
    adv_data_dict = pickle.load(f)

xs = adv_data_dict["images"]
y_trues = adv_data_dict["y_trues"]
noises  = adv_data_dict["noises"]
y_preds_adversarial = adv_data_dict["y_preds_adversarial"]

# Visualize N random images
idxs = np.random.choice(range(100), size=(30,), replace=False)
for matidx, idx in enumerate(idxs):
    orig_im = xs[idx].reshape(3, 32, 32).transpose([1, 2, 0])
    adv_im  = orig_im + noises[idx].reshape(3, 32, 32).transpose([1, 2, 0])
    noise_im = noises[idx].reshape(3, 32, 32).transpose([1, 2, 0])
    
    orig_im = np.array(orig_im); adv_im = np.array(adv_im); noise_im = np.array(noise_im)
    orig_im = (orig_im - orig_im.min())/(orig_im.max() - orig_im.min())
    adv_im = (adv_im - adv_im.min())/(adv_im.max() - adv_im.min())
    noise_im = (noise_im - noise_im.min())/(noise_im.max() - noise_im.min())
    
    disp_im = np.concatenate((orig_im, adv_im, noise_im), axis=1)
    plt.subplot(5, 6, matidx+1)
    #plt.imshow(disp_im, "color")
    plt.imshow(disp_im)
    plt.xticks([])
    plt.yticks([])
plt.show()

# Noise statistics
noises, xs, y_trues = np.array(noises), np.array(xs), np.array(y_trues)
noises = noises.squeeze(axis=1)
xs = xs.squeeze(axis=1)
adv_exs = xs + noises
print "Adv examples: max, min: ", adv_exs.max(), adv_exs.min()
print "Noise: Mean, Max, Min: "
print np.mean(noises), np.max(noises), np.min(noises)
