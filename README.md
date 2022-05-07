## Requirements
Python with Anaconda distribution and jupyter.

The following packages are needed:
 - ``openCV 4.5.5`` or higher
 - ``matplotlib``
 - ``scipy``
 - ``numpy``

Create the folder ``images/results`` to save the output images.

Project folder structure:
```
cv_product_recognition_project
├── images
│   ├── models
│   │   ├── 0.jpg
│   │   ├── 10.jpg
│   │   ├── 11.jpg
│   │   ├── 12.jpg
│   │   ├── 13.jpg
│   │   ├── 14.jpg
│   │   ├── 15.jpg
│   │   ├── 16.jpg
│   │   ├── 17.jpg
│   │   ├── 18.jpg
│   │   ├── 19.jpg
│   │   ├── 1.jpg
│   │   ├── 20.jpg
│   │   ├── 21.jpg
│   │   ├── 22.jpg
│   │   ├── 23.jpg
│   │   ├── 24.jpg
│   │   ├── 25.jpg
│   │   ├── 26.jpg
│   │   ├── 2.jpg
│   │   ├── 3.jpg
│   │   ├── 4.jpg
│   │   ├── 5.jpg
│   │   ├── 6.jpg
│   │   ├── 7.jpg
│   │   ├── 8.jpg
│   │   └── 9.jpg
│   └── scenes
│       ├── e1.png
│       ├── e2.png
│       ├── e3.png
│       ├── e4.png
│       ├── e5.png
│       ├── h1.jpg
│       ├── h2.jpg
│       ├── h3.jpg
│       ├── h4.jpg
│       ├── h5.jpg
│       ├── m1.png
│       ├── m2.png
│       ├── m3.png
│       ├── m4.png
│       └── m5.png
├── product_recognition.ipynb
├── README.md
├── report
│   └── product-recognition-on-store-shelves.pdf
├── utils
│   ├── bbox_filtering.py
│   ├── __init__.py
│   ├── matchers.py
│   ├── plot_clusters.py
│   └── visualization.py
├── weights
│   ├── EDSR_x4.pb
│   ├── ESPCN_x4.pb
│   ├── FSRCNN-small_x4.pb
│   └── LapSRN_x4.pb
└── workflow.ipynb
```
