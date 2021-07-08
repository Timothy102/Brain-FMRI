# AGD-Autoencoder: Attention Gated Deep Convolutional Autoencoder for Brain Tumor Segmentation

## Abstract

Brain tumor segmentation is a challenging problem in medical image analysis. The endpoint is to generate the salient masks that accurately identify brain tumor regions in an fMRI screening. In this paper, we propose a novel attention gate (AG model) for brain tumor segmentation that utilizes both the edge detecting unit and the attention gated network to highlight and segment the salient regions from fMRI images. This feature enables us to eliminate the necessity of having to explicitly point towards the damaged area(external tissue localization) and classify(classification) as per classical computer vision techniques. AGs can easily be integrated within the deep convolutional neural networks(CNNs). Minimal computional overhead is required while the AGs increase the sensitivity scores significantly. We show that the edge detector along with an attention gated mechanism provide a sufficient enough method for brain segmentation reaching an IOU. 

Read the full article on Arxiv: https://arxiv.org/abs/2107.03323. 

## Requirements

* pandas==1.1.4
* numpy==1.19.5
* mne==0.22.0
* matplotlib==3.3.3
* glob2==0.7
* datetime==4.3

`
pip3 install -r requirements.txt
`

## Model training and Inference

We used a k-fold cross-validation method to test the network performance. The approach we used randomly divides the data into 10 approximately equal batches, to provide the randomness factor. This is something also referred to as the record-wise-cross-validation. was implemented to test the generalization capability of the network in medical diagnostics [25].
The generalization capability in clinical practice represents the ability to predict the diagnosis based
on the data obtained from subjects from which there are no observations in the training process.
## Utilizing the Model

The idea of providing edge information for further processing is not a new idea. Deriving from the one ..., to the Inf-Net module they have shown great results to provide constraints to guide feature extraction for segmentation. 

In order to learn the edge representation, we feed the low-level feature with moderate resolution to the proposed edge attention [EA] module to a convolutional 1x1 filter to produce the filter mapping of the original image. The gated module is trained using the standard Binary Cross Entropy. 

We can explore the benefits of AGs for medical imaging in the context of image segmentation by proposing a grid-based gating that allows attention coefficients to be more specific to local regions.


## Results and Discussion

Results of the developer AGD Autoencoder are shown in Table 2 and visualized using the confusion matrices, as shown in Figure 3. We trained the model using weighted Focal Loss Binary Cross Entropy. In order to neglect the imbalance of classes and tumors in the database, we have also shown mean, IOU, precision, recall and F1-Score.

Confusion matrices for the subject-wise 10-fold cross-validation approach for testing data from the augmented dataset are shown in Figure 5. The IOU for the testing data stands at 81 percent. 

## Contact and Acknowledgements

I would like to thank the Physionet's Sleep EDF community for the open dataset upon which my research was concluded. Feel free to contact me with any proposals on LinkedIn: linkedin.com/in/tim-cvetko-32842a1a6/ or Mail: cvetko.tim@gmail.com :D

### Citation

@misc{cvetko2021agdautoencoder,
      title={AGD-Autoencoder: Attention Gated Deep Convolutional Autoencoder for Brain Tumor Segmentation}, 
      author={Tim Cvetko},
      year={2021},
      eprint={2107.03323},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
