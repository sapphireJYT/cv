EN.600.661 HW1

Part 1

1. The threshold value used for p1 function is 120.

2. Additional properties for p3 function:
    I added 'area' feature, the approximate area of the object.
    I also added the 'radius' feature, which is the square root of the object area:
        radius = sqrt(area / pi)

    The radius feature is then used as the radius of the circle drawn for each object, so that the circle size would be proper for each object, neither too large nor too small.

3. For p4 function:
    Threshold: 120

    Comparison criteria: 
        Compare the following two properties
            1). min_moment
            2). roundness

    using the equation:
        diff = abs(target_value - obj_value) / obj_value

    If diff < 0.2 for both properties, then mark the object as one of the targets.


Part 2

1. Convolution masks:
I tried Roberts masks, Prewitt masks, 3x3 Sobel and 5x5 Sobel masks as given in the slide. Prewitt masks give the worst result after thresholding. 5x5 Sobel tends to yield stronger lines than 3x3 Sobel after thresholding.

I chose 3x3 Sobel masks so that edge points are sufficient but not as many as the 5x5 Sobel, in order to make the Hough transformation faster. 

2. Parameter ranges:
    theta range: [-pi/2, pi/2) (step size pi/180)
    rho range is a vector of N evenly spaced values between [rho_min, rho_max), where 
        rho_max = image diagonal
        rho_min = -image diagonal
        N = image diagonal * 2

Though these two ranges cover a lot of values and can be computationally expensive,  they tend to give better results than other coarser parameter ranges I tried.

3. Hough array resolution: 
    num(theta) * num(rho) = 180 * 1592 (Image diagonal is about 796)

Note that the image size after convolution is (478, 638), and the edge thresholded image is used as the input image_in for p7 function. The detected lines would then be drawn above the edge thresholded image.

I chose these resolution to make sure all the parameter values will be considered in the voting.
    
Voting Scheme:
    To reduce computation, first extract all edge points, then go through each possible theta value (180 thetas in total), compute the corresponding rho using the equation:
        x * sin(theta) - y * cos(theta) + rho = 0

    Choose the closest rho from the rho range and add one to the accumulator[t][r] where t and r are the corresponding indices of the theta and rho we just computed. Last, scale the accumulator down to [0, 255].

Edge Threshold in p6: hough_simple_1.pgm: 40

4. Accumulator threshold for p7:
    hough_simple_1.pgm: 164
    hough_simple_2.pgm: 132
    hough_complex_1.pgm: 130

The detection for hough_complex_1.pgm would look much better if viewed after pruning the lines with p8 function using threshold 60.

5. End-point Detection:
    First threshold the input hough image and plot infinite lines as p7 does. Here let old_output denote the resulted image.

    Given the edge thresholded image and the hough transformed output image: 

    for each infinite line in the HT output image(denoted as old_output here):
        for each edge point (x, y) on the line:
            if edge_thresholded[x][y] > 0:
                Set cropped_out[x][y] = old_output[x][y]

    Therefore, only the edge points thresholded out before the Hough Transformation will remain in the cropped output image. 

Accumulator threshold for p8:
    hough_simple_1.pgm: 164
    hough_simple_2.pgm: 132
    hough_complex_1.pgm: 60

Note: the red lines are the detected cropped lines.
