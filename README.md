# web interface to run cellpose

[![launch ImJoy](https://imjoy.io/static/badge/launch-imjoy-badge.svg)](https://imjoy.io/#/app?plugin=https://cellpose.org)

If you'd like to run it on your local computer, run
~~~
pip install -r requirements.txt
python main.py
~~~

Then open a web browser and go to the website that it specifies. This version does NOT use the GPU and only works for 2D images. It also resizes the images to 512 max on one side.

Sample images are not saved in this repo so that functionality will not work.

## Using CellPose.org as an ImJoy plugin

This web application supports `imjoy-rpc`-- a protocol enables remote function calls beween ImJoy plugins.

This means you can use cellpose.org as an ImJoy plugin and passing images into it for segmentation.

Basically, you will be able to show an the cellpose window and call `segment` function in Python or Javascript:
``` python
cellpose = await api.createWindow(src="https://cellpose.org")

image = ...
# The `segment` function accepts the following keys:
# - input: a numpy array contains the input image with 1 or 3 channels, uint8 or uint16
# - diam: the average cell `diameter` in pixels
# - net: network type,  cyto or nuclei
# - chan1: cytoplasm channel, grayscale=0, R=1, G=2, B=3
# - chan2: nuclei channel, grayscale=0, R=1, G=2, B=3
# - outputs: possbile values are geojson,mask,flow,img,outline_plot,overlay_plot,flow_plot,img_plot
result = await cellpose.segment({"input": image, "diam": 30, "net": "cyto", "chan1": 1, "chan2": 3, "outputs": "flow,mask,outline_plot,overlay_plot"})
```

You can find a live demo [here](https://ij.imjoy.io/?plugin=https://gist.github.com/oeway/c9592f23c7ee147085f0504d2f3e993a) with its [source code in Javascript](https://gist.github.com/oeway/c9592f23c7ee147085f0504d2f3e993a). For Python, [here](https://gist.github.com/oeway/cec7b38e0a8fcda294de5362c07072f0) is an example notebook.



