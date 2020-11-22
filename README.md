# web interface to run cellpose

[![launch ImJoy](https://imjoy.io/static/badge/launch-imjoy-badge.svg)](https://imjoy.io/#/app?plugin=https://cellpose.org) [![open in ImageJ.JS](https://ij.imjoy.io/assets/badge/open-in-imagej-js-badge.svg)](https://ij.imjoy.io/)

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

Basically, you will be able to show an the cellpose window and call `segment` function in Javascript or Python:
``` python
    cellpose = await api.createWindow(src="https://cellpose.org")

    # read an image as numpy array
    image = np.random.randint(0, 255, [500, 500], dtype='uint8')

    # The `segment` function accepts the following keys:
    # - input: an image/array contains the input image with 1 or 3 channels, uint8 or uint16, in python you can pass an numpy array or an URL, in javascript, you can pass an File object, an URL, or an base64 encoded png image.
    # - diam: the average cell `diameter` in pixels
    # - net: network type,  cyto or nuclei
    # - chan1: cytoplasm channel, grayscale=0, R=1, G=2, B=3
    # - chan2: nuclei channel, grayscale=0, R=1, G=2, B=3
    # - invert: invert the input image before feeding into cellpose
    # - keep_size: keep the output size (same as input)
    # - outputs: possbile values are geojson,mask,flow,img,outline_plot,overlay_plot,flow_plot,img_plot
    # the results is a dictionary or object containing different output types depending on the `outputs` argument, e.g. you will get a geojson object if you pass `geojson` in the `outputs` key. For Python if you pass an numpy array, then the returned mask and flow will also be encoded as an numpy array, otherwise, all the images will be saved in png format and encoded as base64 string.
    result = await cellpose.segment({"input": image, "diam": 30, "net": "cyto", "chan1": 1, "chan2": 3, "outputs": "flow,mask,outline_plot,overlay_plot"})

    # we will get an numpy array jere
    print(result["mask"])
```

CellPose is also integrated with ImageJ.JS, you can [![launch ImageJ.JS](https://ij.imjoy.io/assets/badge/launch-imagej-js-badge.svg)](https://ij.imjoy.io), then run it by clicking "Segment with CellPose" in the plugin menu. The source code of the plugin(in JavaScript) is [here](https://gist.github.com/oeway/c9592f23c7ee147085f0504d2f3e993a). For Python, [here](https://gist.github.com/oeway/cec7b38e0a8fcda294de5362c07072f0) is an example notebook.

