<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Mentor][mentor-shield]][mentor-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

## Panoptic segmentation using DETR transformers
________

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Prerequisites](#prerequisites)
* Dataset Preparation(#Dataset-Preparation)
* [Approach](#Approach)
* [Code](#Code)
* [Network Diagram](#Network-Diagram)
* [Results](#Results)
* [License](#license)
* [Mentor](#mentor)

## Prerequisites

* [Python 3.8](https://www.python.org/downloads/) or Above
* [Pytorch 1.8.1](https://pytorch.org/)  
* [Google Colab](https://colab.research.google.com/)

<!-- Dataset-Preparation -->
## Dataset Preparation

	1) Pass the images we collected through pre-trained DETR panoptic network to get masks.
	2) Pre-trained DETR network is trained on Coco dataset which wont be having the "things" we are having for capstone
	3) But it can predict the "stuff" present in our image
	4) It will also predict certain objects as "things" in our image. eg: Aeroplane. But these are not the "things" we are interested in.
	5) So we will treat these predicted "things" as "misc_stuff" for our problem
	6) Additionally, we have 2 problems:
		a. Capstone "Things" in our image, could get detected and segmented as something else. Eg: Brick as suitcase. We want to avoid 
		this
		b. "Stuff" in our annotated dataset could be very limited. Our images are mainly focussed on construction sites. So background 
		will be mostly similar. This will limit the ability of our panoptic model to identify "stuff" correctly
	7) To solve pblm (a), we will mask the objects we annotated from our original image and then pass this image to DETR panoptic network.
	8) This means we are passing only images that has stuff and misc_stuff only to DETR pre-trained panoptic network.
	9) Thus, DETR pre-trained output will give us masks and segmentation coordinates for "stuff" and "misc_stuff".
	10) Images we annotated are having segmentation coordinates of "things" we are interested for this capstone.
	11) Thus, we can combine segmentation coordinates from step 9 and step 10. This will give us segmentation coordinates for "stuff", 
	"misc_stuff" and "things" corresponding to our original image.
	12) Now, we can generate the bounding boxes for "stuff" and "misc_stuff" from the masks we got from step 9 via below code base.
	https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#contour-features
	13) We can generate the bounding boxes for "things" from the step 7 images.  In step 7 where we masked our things to avoid 
	mis-representation before passing to DETR pre-trained panoptic code.
	14) We can combine BB from step 12 & step 13. This will give us BB coordinates for "stuff", "misc_stuff" and "things" corresponding 
	to our original image.
	15) Thus from step 11 and step 14, we have below items for our dataset
			i. Original image
			ii. Stuff, Misc_Stuff & things as applicable in the image
			iii. BB coordinates & segmentation coordinates for each of these
	16) To solve pblm (b), we will take Coco stuff test dataset which will be having a healthy amount of images having stuff in it. 
	17) We will pass these images through DETR pretrained panoptic code to get masks for "stuff" and "misc_stuff".
	18) Then with these masks we will get BB for these "stuff" and "misc_stuff".
	19) Coco test dataset labels will already be having segmentation coordinates. Now, we have BB also from step 18.
	20) Take dataset from 19 and combine it with dataset from 15. This will give us our final dataset. 

<!-- Approach -->
## Approach

- Training
    - Overall panoptic segmentation architecture is as below. Details of each of these steps are as explained below

    ![Overall_Arch](https://github.com/anilbhatt1/Panoptic_Segmentation_DETR/blob/master/Readme_Images/Overall_Panoptic_Arch.jpg)

    - Object detection using DETR network diagram is as below

    ![DETR_BB](https://github.com/anilbhatt1/Panoptic_Segmentation_DETR/blob/master/Readme_Images/DETR_BB_Architecture.png)
    
    - Now let us get into the details of panoptic segmentation approach.
    - Get pre-trained ResNet-50. Strip out GAP & final layer.
    - Pass the batch of images through ResNet-50.
    - Get the **intermediate activations** out of ResNet-50. Res-net 50 has a stride of 32. 
    - For example, if we give a 480x480 image (min resolution required for DETR to work), we will get back 15x15x2048. This will be flattened to 225x256.
    - Pass these activations along with positional encoding to encoder. Also set aside these activations, we will need these for **binary map generation** later.
    - Positional encodings will be sine based.
    - Encoder-decoder architecture is as below.
    
    ![En_De](https://github.com/anilbhatt1/Panoptic_Segmentation_DETR/blob/master/Readme_Images/DETR_BB_Encoder_Decoder_Architecture.png)

    - Encoder will provide an attention matrix.
    - Role of encoder is to separate the object instances by giving high attention scores to the pixels belonging to the same object. 
    - Encodings coming out of encoder will be similar to below.
    
    ![Encodings](https://github.com/anilbhatt1/Panoptic_Segmentation_DETR/blob/master/Readme_Images/Encoding_Results.png)
    
    - Decoder will have multiple layers. 
    - Last decoder layer helps us to find bounding box & class.
    - Decoder also uses attention to find out the objects as shown below. 
    
    ![Decoder_Attn](https://github.com/anilbhatt1/Panoptic_Segmentation_DETR/blob/master/Readme_Images/Decoder_Attn_Mpa.png)
    
    - Attention focuses on the extremities of each object to find out the bounding boxes for corresponding objects as shown below

    ![Decoder_Attn_BB](https://github.com/anilbhatt1/Panoptic_Segmentation_DETR/blob/master/Readme_Images/Decoder_Attn_BB.png)
    
    - Object queries are the input to decoder. 
    - In the problem in hand, i.e. panoptic segmentation, role of object query is to scan & find the object from encodings we get from encoder.
    - There will be 100 Object queries in DETR. Shape will be 100x256 i.e. Each object query will be having a 256 dimension.
    - Below is a sample image that shows object centers predicted by these 100 queries.
    
    ![Obj_Q1](https://github.com/anilbhatt1/Panoptic_Segmentation_DETR/blob/master/Readme_Images/Object_query_preds.png)
    
    - Below image shows which query was active while finding the object (giraffe) on left side.
    
    ![Active_OQ1](https://github.com/anilbhatt1/Panoptic_Segmentation_DETR/blob/master/Readme_Images/Active_oq_1giraffe.png)
    
    - Below image shows the active queries (coloured dots) responsible for finding the objects. 
    - We can see 24 queries active (colour dots on right) corresponding to 24 images (left)!

    ![Active_OQ2](https://github.com/anilbhatt1/Panoptic_Segmentation_DETR/blob/master/Readme_Images/Active_oqs_multi_giraffe.png)
    
    - Panoptic segmentation is a combination of instance segmentation (bb with classes) & semantic segmentation (background classes) as shown below.
    
    ![Panoptic_seg](https://github.com/anilbhatt1/Panoptic_Segmentation_DETR/blob/master/Readme_Images/Panoptic_Segm.png)
    
    - We train the DETR to get boxes around things & stuff.
    - After decoder we will have **one object embedding for each object** as in below image (cow - thing, grass, trees, sky - stuff).
    
    ![Decoder_op](https://github.com/anilbhatt1/Panoptic_Segmentation_DETR/blob/master/Readme_Images/Decoder_Op.png)
    
    - We will then use a multi-head attention over the **encoded image** to return the **attention maps** for **each of the object embedding**.
    
    ![Attn_map_OE](https://github.com/anilbhatt1/Panoptic_Segmentation_DETR/blob/master/Readme_Images/Attention_map_OE.png)
    
    - Next, we will upsample these **attention maps** & clean the masks using **intermediate activations we got from Resnet-50** backbone. 
    - We will get binary masks as a result.
    
    ![Binary_map](https://github.com/anilbhatt1/Panoptic_Segmentation_DETR/blob/master/Readme_Images/Binary_Mask_Creation.png)
    
    - Next we will overlay these masks with original image as below to get segmentation. 
    - Finally, using pixel-wise argmax we will decide which class each mask belongs to.
    
    ![Segment](https://github.com/anilbhatt1/Panoptic_Segmentation_DETR/blob/master/Readme_Images/Overlay.png)

<!-- Code -->
## Code

<!-- Network-Diagram-->
## Network Diagram
- Network Diagram is as below

<!-- Results -->
## Results

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- MENTOR -->
## Mentor

* [Rohan Shravan](https://www.linkedin.com/in/rohanshravan/) , [The School of A.I.](https://theschoolof.ai/)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[mentor-shield]: https://img.shields.io/badge/Mentor-mentor-yellowgreen
[mentor-url]: https://www.linkedin.com/in/rohanshravan/
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat-square
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/anilbhatt1/Deep_Learning_EVA4_Phase2/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555




