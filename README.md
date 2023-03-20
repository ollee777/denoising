#Denoising
Code for NTIRE 2023 Image Denoising Challenge

#Test Instruction
Please put 'BenchmarkNoisyBlocksSrgb.mat' under 'data/' folder.
python test.py --name RDN_e40_16 --which_model final_net.pth --test_path data/BenchmarkNoisyBlocksSrgb.mat
