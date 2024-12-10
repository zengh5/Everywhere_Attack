# Everywhere_attack
Codes for our paper to 2025AAAI: H. Zeng, S. Cui, B. Chen, and A. Peng, 'Everywhere Attack: Attacking Locally and Globally to Boost Targeted Transferability.' 

The proposed method can be illustrated with the following figure. To fool a DNN model to misclassify a 'Bajie' image as 'Wukong', we plant an army of 'Wukong's to the 'Bajie'. Specifically, we split the 'Bajie' image into non-overlap blocks abd jointly mount a targeted attack on each block. Such a strategy avoids transfer failures caused by attention inconsistency between surrogate and victim models and thus results in strong transferability.  
<div align=center>
<img src="fig/Fig1.png" width="750">
</div>

## Usage
Please run everywhere_demo.py to see the targeted transferability improvement by the proposed _everywhere_ method.

### Acknowledgement
Our implementation is highly borrowed from [Zhao's code on NeurIPS 2021](https://github.com/ZhengyuZhao/Targeted-Transfer).
