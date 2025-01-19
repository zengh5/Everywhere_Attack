# Everywhere_attack
Codes for our paper to 2025AAAI: H. Zeng, S. Cui, B. Chen, and A. Peng, 'Everywhere Attack: Attacking Locally and Globally to Boost Targeted Transferability.'  [arXiv](https://arxiv.org/abs/2501.00707)

The proposed method can be illustrated with the following figure. To fool a DNN model to misclassify a 'Bajie' image as 'Wukong', we plant an army of 'Wukong's to the 'Bajie'. Specifically, we split the 'Bajie' image into non-overlap blocks and jointly mount a targeted attack on each block. Such a strategy avoids transfer failures caused by attention inconsistency between surrogate and victim models and thus results in strong transferability.  
<div align=center>
<img src="fig/Fig1.png" width="750">
</div>

## Usage
Please run everywhere_demo.py to see the targeted transferability improvement by the proposed _everywhere_ method.
If you want to get the SOTA result, please try CFM+everywhere attack with 'everywhere_CFM_10tar_github.py'. Note, you may need to download the NIPS2017 dataset to the 'dataset' folder first.

### Acknowledgement
Our implementation is highly borrowed from [Zhao's code](https://github.com/ZhengyuZhao/Targeted-Transfer) on NeurIPS 2021.
