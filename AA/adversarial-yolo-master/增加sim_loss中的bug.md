
```bash
C:\conda\envs\CUDA110_torch\python.exe C:\Users\lutao\Desktop\git_AA_NAS\AA\adversarial-yolo-master\train_patch_sim_Loss.py A4RealWorld 
C:\conda\envs\CUDA110_torch\lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and will be removed in 0.15, please use 'weights' instead.
  warnings.warn(
C:\conda\envs\CUDA110_torch\lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and will be removed in 0.15. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
GPU-cuda:1 for evaluate. Use ssim for metric 
56 BS
img_size 416 batch_size 56 n_epochs 5000 max_lab 14
One epoch is 11
TensorFlow installation not found - running with reduced feature set.
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.13.0 at http://localhost:6006/ (Press CTRL+C to quit)
Running epoch 0:   0%|          | 0/11 [00:00<?, ?it/s]C:\Users\lutao\Desktop\git_AA_NAS\AA\adversarial-yolo-master\train_patch_sim_Loss.py:121: UserWarning: Anomaly Detection has been enabled. This mode will increase the runtime and should only be enabled for debugging.
  with autograd.detect_anomaly():
C:\conda\envs\CUDA110_torch\lib\site-packages\torch\nn\functional.py:4277: UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0. Please specify align_corners=True if the old behavior is desired. See the documentation of grid_sample for details.
  warnings.warn(
C:\conda\envs\CUDA110_torch\lib\site-packages\torch\nn\functional.py:4215: UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0. Please specify align_corners=True if the old behavior is desired. See the documentation of grid_sample for details.
  warnings.warn(
Running epoch 0:   0%|          | 0/11 [00:12<?, ?it/s]
Traceback (most recent call last):
  File "C:\Users\lutao\Desktop\git_AA_NAS\AA\adversarial-yolo-master\train_patch_sim_Loss.py", line 290, in <module>
    main()
  File "C:\Users\lutao\Desktop\git_AA_NAS\AA\adversarial-yolo-master\train_patch_sim_Loss.py", line 287, in main
    trainer.train()
  File "C:\Users\lutao\Desktop\git_AA_NAS\AA\adversarial-yolo-master\train_patch_sim_Loss.py", line 158, in train
    sim_score = self.sim_scorer.get_sim_score(tensor1, tensor2)
  File "C:\Users\lutao\Desktop\git_AA_NAS\AA\adversarial-yolo-master\tools\simlarity\sim_calculate.py", line 90, in get_sim_score
    score_sim = self.cal_sim(fea1,fea2)
  File "C:\Users\lutao\Desktop\git_AA_NAS\AA\adversarial-yolo-master\tools\simlarity\sim_calculate.py", line 207, in cal_sim
    result.append(self.value_calculator(img1, img2))
  File "C:\conda\envs\CUDA110_torch\lib\site-packages\image_similarity_measures\quality_metrics.py", line 222, in ssim
    return structural_similarity(org_img, pred_img, data_range=max_p, channel_axis=2)
  File "C:\conda\envs\CUDA110_torch\lib\site-packages\skimage\metrics\_structural_similarity.py", line 133, in structural_similarity
    ch_result = structural_similarity(im1[_at(ch)],
  File "C:\conda\envs\CUDA110_torch\lib\site-packages\skimage\metrics\_structural_similarity.py", line 178, in structural_similarity
    raise ValueError(
ValueError: win_size exceeds image extent. Either ensure that your images are at least 7x7; or pass win_size explicitly in the function call, with an odd value less than or equal to the smaller side of your images. If your images are multichannel (with color channels), set channel_axis to the axis number corresponding to the channels.

Process finished with exit code 1

```