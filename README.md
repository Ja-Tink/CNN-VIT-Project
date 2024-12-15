# CNN-VIT-Project
Lana Lee, Alex Manko, Jack Tinker Advanced Topics Project

This project explores methods to combine the stregnths of CNNs and the more modern vision transformers. We used the Food-101 dataset, introduced in [Bossard, L, et al.](https://link.springer.com/chapter/10.1007/978-3-319-10599-4_29), to evaluate the efficacy of our approachs.

`CODE.ipynb` contains all code for this project, including data loading and preprocessing, model definition, training, and evaluation. Each section is clearly labelled.

PyTorch support for transformers is currently very limited relative to CNNs, so we began by implementing a basic customizable ViT architecture from scratch. Debugging/preliminary evaluation of this model was performed using the MNIST dataset. This simple model failed to perform on the Food-101 dataset, likely due to the relatively small size of Food-101. In light of this we pivoted to a ViT pretrained on ImageNet1K to get a baseline performance. Setting the last 4 of 16 attention heads to be trainable gave the following results:

The pretrained vision transformer reached around 60% accuracy before rapidly overfitting.

<img width="300" alt="PNG image" src="https://github.com/user-attachments/assets/7f6cd3d6-7a55-4ab8-9a0a-e6670bc7c076" />

We then tried a naive combination of pretrained ViT and ResNet50 (also pretrained on ImageNet) by simply concatenating their outputs and passing it through a linear layer. Layers were frozen to only train this final linear layer.

<img width="250" alt="combinedAcc" src="https://github.com/user-attachments/assets/5dbd53e1-4dfe-4477-a78f-224b3d4a91d1" />
<img width="250" alt="combinedLoss" src="https://github.com/user-attachments/assets/b0561bf9-46b3-48f2-a55a-21d9cea6ad01" />

Next we used ResNet50 as a feature exactor to pass values to our untrained ViT. ResNet was frozen and the ViT was trained.

<img width="300" alt="AGV_vUczCidwzkSzsXZ2Wlofyo3ITEgA5-nCWdPBPNnvb0j2GwZrdMuWJR3yQ13wI4JA-Ylbo5echRkHRsXvywDCcVvbJf-vibStAhbToc2HxjGT5A2ckFP7br9l3hlRLwLK_2CHqtDY0w=s2048" src="https://github.com/user-attachments/assets/c949e784-dafc-4f4c-8a78-77e714a79f86" />

Finally we ran the two pretrained models in parallel, frozen like before, but with an attention mechanism before the linear layer to attend to the outputs and weight them. The results of this model are below:

<img width="300" alt="AGV_vUf7U-VsuXAAXvxZiLKDHzOslKs4jRVhsnmyNOb_0gAvZ5a2UWL3VaiF5nHYjtmT3hiihI_uZgZACKZI8AULwP4d_3inVlfqzLTT6FKG3n3JjzWWIy-VYzvfc6xDKcFAuTprr3IqsQ=s2048" src="https://github.com/user-attachments/assets/6ac49b02-2a47-4d2a-82a0-9b6848ff0953" />

Below are sample predictions produced by our final model. (Code to produce such results can be found at the end of the notebook)
<img width="984" alt="Screenshot 2024-12-14 at 10 08 11 PM" src="https://github.com/user-attachments/assets/eccff595-7be2-4d0b-ae57-30af99c9bcb7" />

Given more time and computing resources, we would like to further explore how performance of the combined model can be improved by unlocking the later layers of the pretrained models for finetuning. Another promising approach would be to use these hybrid models to apply pseudolabels to an unlabeled dataset for student-teacher learning. Vision Transformers are an inherently data hungry architecture, so more data would likely greatly improve performance, especially in the case of training from scratch.
