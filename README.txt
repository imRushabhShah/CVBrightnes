##################################################################
All requirements are in the requirements.txt file
just execute 
pip install -r requirements.txt

##################################################################
Files Description

Luv_LinearStretching.py :Brightness correction using linear streaching 
on L value in Luv

to execute 
python Luv_LinearStretching.py 0 0.2 0.8 0.8 test-image.bmp out1a.png 


LUV_histogram_equalization.py :Brightness correction using Histogram 
Equalization on L value in Luv

to execute 
python LUV_histogram_equalization.py 0 0.2 0.8 0.8 test-image.bmp out1b.png 


xyY_linear_Stretching.py :Brightness correction using linear streaching 
on Y value in xyY

to execute 
python xyY_linear_Stretching.py 0 0.2 0.8 0.8 test-image.bmp out1c.png 


#################################################################

Assumtions:

Whenever encountered division by zro the variable value 
was assumed to be zero.

If the varible value was going out of bound it was pushed
to the nearest boundary point
For instance L value was restrained to be between 0 to 100

Every variables were converted to float/ double to have 

################################################################

Bad Images are seen in Histogram equilization when the 
white spots are scean in the image. From the analysis made 
on various other images the best result is on Images with high 
contrast colors
precision upto 6 to 8 decimal places.

Execution of each program displays the image and also stores 
a copy of altered image.
