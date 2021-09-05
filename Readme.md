## Data

Data used or created within this library can be downloaded at
https://mega.nz/folder/SyAz2SJZ#Z8nLESRrYAetIYFOFLscKg

#Modifications to the original library

The original library can be found here: (https://github.com/acse-jy220/SFC-CAE-Ready-to-use)

To use the SFC-CAE for different Space Filling Curves, several changes were made to the original library.

Firstly, the Encoder and Decoder object creation was made "modular", with several methods: 

- set_sfcs() with argument spacefillingorderings, a list of space-filling curves which sets the sfcs. Essential for shuffling curves.

- set_parameters() with optional arguments kernelsize, stride, increasemulti, numfinalchannels, and activation 

- find_layers() and setmodules()) which respectively find out what the structure of the autoencoder (in terms of layers, convolutions and fully connected layer sizes etc) and set the PyTorch modules in the right places

I have also introduced various options for slight modifications to the architecture of the SFC-CAE:

I have added an argument called nfclayers with which one can decide the number of fc layers to put between the convolutional parts of the autoencoder.
The argument is used in find_layers() to find out the structure of the autoencoder.

An option was implemented to add "smoothing layers" with stride 1 and custom channels and kernel size before or after the autoencoder's structure.
It is hoped that these "smoothing layers" would be able to better extract features from the input or "smooth" the output with more filters.

the smoothing layers parameter is a list of two lists, the smoothing layers that come before the encoder and those that come after the encoder:

smoothinglayers = [[],[]] would produce no smoothing layers
smoothinglayers = [[(8,33)],[]] would produce 1 smoothing layer before the encoder with 8 channels and kernel size 33
smoothinglayers[[],[(64,17),(32,9)]] adds two smooting layers after the decoder with 64 and 32 channels and kernel size 17 and 9 respectively.

Remember: kernel size HAS to be odd!

An option was also implemented to feed coordinates to convolutional layers within the autoencoder.

If provided, coordinates for the data are stored within the autoencoder object.
If the user has asked to feed these coordinates to a certain convolutional layer, the coordinates are reordered according to the current sfc ordering, then "coarsened" according to the size of the data within the convolutional layer in question.
By "coarsened" I mean that if the size of the input of a certain convolutional layer is 1000 and I have 4000 coordinates, I will choose every 4th coordinate to create a sort of "coarsened" space-filling curve.
They are then appended as ulterior channels to the data, and the next convolutional layer has an additional 2 channels (x and y) of input.

There are two options as well: one can choose to feed the coordinates as they are or one can decide to feed their difference. By difference I mean the x and y distance between the previous and the next node in the space-filling curve ordered mesh, which is 4 values for each node and therefore adds 4 channels to the input of the next layer.

coords = coords, 		#Gives the autoencoder the coordinates
coption = 1,     		#Sets the coordinate option to 1 (ensuring that the actual coordinates will be fed to layers instead of the distance of each point from its neighbours)
coordslayers = [0,1]	#The coordinates will be fed to no layers in the encoder and to the the last layer in the decoder
coordslayers = [2,3]    #The coordinates will be fed to the first two layers in the encoder and the last 3 layers of the decoder

Coordinate "feeding" was also implemented for the fully connected layers, although on a more limited scale (if the feedcordsfc option is turned on, it will only feed coordinates to the first of the fully connected layers).
The fcoption works similarly to the standard coption: one can choose whether to feed the actual coordinates or the distance representation.

One of our ideas was to append two space filling curves to each other. I quickly realised that there was no difference between appeding two space filling curves to each other and using the same filter on two different ones, save for avoiding the jump in the middle.
The samefilter = True option does just that

Among other things, a verbose option was added to check the inner workings of the SFC-CAE.

verbose = True will show the size of the tensor as it goes through the autoencoder.

paramlist = [kernel_size, increase_multi, stride, num_final_channels, activation] creates an autoencoder with the parameters provided, else it defaults to whatever is preset

Two other classes were also created, %SFC_CAE_Adaptive and SFC_CAE_Interpol (short for interpolation), which are used for the adaptive decoder padding and the interpolation autoencoder respectively.
