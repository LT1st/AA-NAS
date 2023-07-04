BUG:
Codes/AA and NAS/把人物抠出来 获取背景.py

图像经过mask操作后，出现了反转负片的效果
```
        tensor = torch.from_numpy(mask).unsqueeze(0)  # 将NumPy数组转换为PyTorch张量，并添加通道维度
        mask_tensor = tensor.repeat(3, 1, 1)  # 将张量复制为形状为(3, 256, 256)的张量
        input_image_masked = input_tensor_nopre * mask_tensor
```
![crop001522.png](background_without_human%2Fclean%2Fcrop001522.png)


![crop001522_background.png](background_without_human%2Fclean%2Fcrop001522_background.png)