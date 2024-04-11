# Elec-378-Final
Project Summary: 
Hey guys, heres a quick summary of the high level ideas of what I've done so you can build off them

First I had to unpack all the audio files all at once from where they were in the directory, I'm not sure how well that line of 
code will translate into colab but we can figure that out.

Then I extracted a bunch of features, I gave them descriptive names I think so its fine

I made the conv_compare function based on the one hw where we convolved with a time reverse signal to see similarity.
I exploit the fact that multiplication in one domain is convolution in other and the fact we already found the fft.
This feature has not been implemented yet since I need to somehow condense this into a number

For now the model I made combines ICA and SVM to work, it was just to get something that works so we can finish
hw7

Then I figured out how to put the predictions into a .csv of the required format.