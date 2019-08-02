# coding=utf-8
# import braindecode
import torch
from torch import nn
from eeggan.modules.layers.reshape import Reshape,PixelShuffle2d
from eeggan.modules.layers.normalization import PixelNorm
from eeggan.modules.layers.weight_scaling import weight_scale
from eeggan.modules.layers.upsampling import CubicUpsampling1d,CubicUpsampling2d
from eeggan.modules.layers.stdmap import StdMap1d
from eeggan.modules.progressive import ProgressiveGenerator,ProgressiveGeneratorBlock,\
							ProgressiveDiscriminator,ProgressiveDiscriminatorBlock
from eeggan.modules.wgan import WGAN_I_Generator,WGAN_I_Discriminator
from torch.nn.init import calculate_gain
class Skip(nn.Module):
	def __init__(self):
		super(Skip).__init__()

	def forward(self, *input):
		return input

class Interpolate(nn.Module):
	def __init__(self, scale_factor, mode):
		super(Interpolate, self).__init__()
		self.scale_factor = scale_factor
		self.mode = mode
		self.interp = nn.functional.interpolate
		
	def forward(self, x):
		x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
		return x

def create_conv_sequence_disc(in_filters,out_filters):
	return nn.Sequential(weight_scale(nn.Conv1d(in_filters,in_filters,9,padding=4),
													gain=calculate_gain('leaky_relu')),
							nn.LeakyReLU(0.2),
							weight_scale(nn.Conv1d(in_filters,out_filters,9,padding=4),
													gain=calculate_gain('leaky_relu')),
							nn.LeakyReLU(0.2),
							weight_scale(nn.Conv1d(out_filters,out_filters,2,stride=2),
													gain=calculate_gain('leaky_relu')),
							nn.LeakyReLU(0.2))
def create_in_sequence_disc(n_chans,out_filters):
	return nn.Sequential(weight_scale(nn.Conv2d(1,out_filters,(1,n_chans)),
													gain=calculate_gain('leaky_relu')),
							Reshape([[0],[1],[2]]),
							nn.LeakyReLU(0.2))
def create_fade_sequence_disc(factor):
	return nn.AvgPool2d((factor,1),stride=(factor,1))

def create_disc_blocks(n_chans, n_out_linear=1):
	blocks = []
	tmp_block = ProgressiveDiscriminatorBlock(
							  create_conv_sequence_disc(50,50),
							  create_in_sequence_disc(n_chans,50),
							  create_fade_sequence_disc(2)
							  )
	blocks.append(tmp_block)
	tmp_block = ProgressiveDiscriminatorBlock(
							  create_conv_sequence_disc(50,50),
							  create_in_sequence_disc(n_chans,50),
							  create_fade_sequence_disc(2)
							  )
	blocks.append(tmp_block)
	tmp_block = ProgressiveDiscriminatorBlock(
							create_conv_sequence_disc(50,50),
							create_in_sequence_disc(n_chans,50),
							create_fade_sequence_disc(2)
							)
	blocks.append(tmp_block)
	tmp_block = ProgressiveDiscriminatorBlock(
							create_conv_sequence_disc(50,50),
							create_in_sequence_disc(n_chans,50),
							create_fade_sequence_disc(2)
							)
	blocks.append(tmp_block)
	tmp_block = ProgressiveDiscriminatorBlock(
							  create_conv_sequence_disc(50,50),
							  create_in_sequence_disc(n_chans,50),
							  create_fade_sequence_disc(2)
							  )
	blocks.append(tmp_block)
	tmp_block = ProgressiveDiscriminatorBlock(
							  nn.Sequential(StdMap1d(),
											create_conv_sequence_disc(51,50),
											Reshape([[0],-1]),
											nn.Linear(50*12,n_out_linear)),
											# weight_scale(nn.Linear(50*12,300),
															# gain=calculate_gain('linear'))),

							  create_in_sequence_disc(n_chans,50),
							  None
							  )
	blocks.append(tmp_block)
	return blocks


def create_conv_sequence_gen(in_filters,out_filters):
	return nn.Sequential(
		# nn.Upsample(mode='linear',scale_factor=2),
							# nn.interpolate(scale_factor=2, mode='linear'),
							Interpolate(scale_factor=2, mode='linear'),
							weight_scale(nn.Conv1d(in_filters,out_filters,9,padding=4),
													gain=calculate_gain('leaky_relu')),
							nn.LeakyReLU(0.2),
							PixelNorm(),
							weight_scale(nn.Conv1d(out_filters,out_filters,9,padding=4),
													gain=calculate_gain('leaky_relu')),
							nn.LeakyReLU(0.2),
							PixelNorm())

def create_out_sequence_gen(n_chans,in_filters):
	return nn.Sequential(weight_scale(nn.Conv1d(in_filters,n_chans,1),
													gain=calculate_gain('linear')),
							Reshape([[0],[1],[2],1]),
							PixelShuffle2d([1,n_chans]))

def create_fade_sequence_gen(factor):
	# return nn.Upsample(mode='bilinear',scale_factor=(2,1))
	return Interpolate(scale_factor=(2,1), mode='bilinear')

def create_gen_blocks(n_chans,z_vars):
	blocks = []
	tmp_block = ProgressiveGeneratorBlock(
								nn.Sequential(weight_scale(nn.Linear(z_vars,50*12),
								gain=calculate_gain('leaky_relu')),
								nn.LeakyReLU(0.2),
								Reshape([[0],50,-1]),
								create_conv_sequence_gen(50,50)),
								create_out_sequence_gen(n_chans,50),
								create_fade_sequence_gen(2)
	)
	blocks.append(tmp_block)
	tmp_block = ProgressiveGeneratorBlock(
								create_conv_sequence_gen(50,50),
								create_out_sequence_gen(n_chans,50),
								create_fade_sequence_gen(2)
								)
	blocks.append(tmp_block)
	tmp_block = ProgressiveGeneratorBlock(
								create_conv_sequence_gen(50,50),
								create_out_sequence_gen(n_chans,50),
								create_fade_sequence_gen(2)
								)
	blocks.append(tmp_block)
	tmp_block = ProgressiveGeneratorBlock(
								create_conv_sequence_gen(50,50),
								create_out_sequence_gen(n_chans,50),
								create_fade_sequence_gen(2)
								)
	blocks.append(tmp_block)
	tmp_block = ProgressiveGeneratorBlock(
								create_conv_sequence_gen(50,50),
								create_out_sequence_gen(n_chans,50),
								create_fade_sequence_gen(2)
								)
	blocks.append(tmp_block)
	tmp_block = ProgressiveGeneratorBlock(
								create_conv_sequence_gen(50,50),
								create_out_sequence_gen(n_chans,50),
								None
								)
	blocks.append(tmp_block)
	return blocks


class Generator(WGAN_I_Generator):
	def __init__(self,n_chans,z_vars, y_vars=8):
		super(Generator,self).__init__()
		# self.pre_concat_layer_x = ProgressiveGeneratorBlock(
		# 	nn.Sequential(
		# 		weight_scale(nn.Linear(z_vars,50*10),
		# 			gain=calculate_gain('leaky_relu')),
		# 		nn.LeakyReLU(0.2),
		# 		Reshape([[0],50,-1]),
		# 		create_conv_sequence_gen(50,50)),
		# 	create_out_sequence_gen(n_chans,50),
		# 	create_fade_sequence_gen(2)
		# 	)
		self.pre_concat_layer_x = nn.Sequential(
			nn.Linear(z_vars, 176),
			nn.LeakyReLU(.2)
		)
		# self.pre_concat_layer_y = ProgressiveGeneratorBlock(
		# 	nn.Sequential(
		# 		weight_scale(nn.Linear(y_vars,50*2),
		# 			gain=calculate_gain('leaky_relu')),
		# 		nn.LeakyReLU(0.2),
		# 		Reshape([[0],50,-1]),
		# 		create_conv_sequence_gen(50,50)),
		# 	create_out_sequence_gen(n_chans,50),
		# 	create_fade_sequence_gen(2)
		# 	)
		self.pre_concat_layer_y = nn.Sequential(
			nn.Linear(y_vars, 24),
			nn.LeakyReLU(.2)
		)

		self.model = ProgressiveGenerator(create_gen_blocks(n_chans,z_vars))
	
	def forward(self, input, y):
		x = self.pre_concat_layer_x(input)
		y = self.pre_concat_layer_y(y)
		# print("x", x.shape)
		# print("y", y.shape)
		x = torch.cat([x, y], 1)
		return self.model(x)

	# def forward(self, input):
	# 	print("shape", self.tmp_block(input).shape)
	# 	return self.model(input)

class Discriminator(WGAN_I_Discriminator):
	def __init__(self,n_chans, n_out_linear=200, y_vars=8):
		super(Discriminator,self).__init__()
		self.model = ProgressiveDiscriminator(create_disc_blocks(n_chans, n_out_linear))
		self.pre_concat_layer_y = nn.Sequential(
			nn.Linear(y_vars, 24),
			nn.LeakyReLU(.2)
		)

		self.pre_concat_layer_x = nn.Sequential(
			nn.Linear(200,  176),
			nn.LeakyReLU(.2)
		)

		self.post_concat_layers = nn.Sequential(
			nn.Linear(200, 150),
			nn.LeakyReLU(.2),
			nn.Linear(150, 100),
			nn.LeakyReLU(.2),
			nn.Linear(100, 50),
			nn.LeakyReLU(.2),
			# nn.Linear(50, 1),
			weight_scale(nn.Linear(50, 1), gain=calculate_gain('linear'))
		)

	def forward(self,input, y):
		x = self.model(input)
		x = self.pre_concat_layer_x(x)
		y = self.pre_concat_layer_y(y)
		x = torch.cat([x, y], 1)
		# print(torch.sum(self.post_concat_layers(x)))
		return self.post_concat_layers(x)
