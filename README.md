# web interface to run cellpose

[![launch ImJoy](https://imjoy.io/static/badge/launch-imjoy-badge.svg)](https://imjoy.io/#/app?plugin=https://cellpose.org)

If you'd like to run it on your local computer, run
~~~
pip install -r requirements.txt
python main.py
~~~

Then open a web browser and go to the website that it specifies. This version does NOT use the GPU and only works for 2D images. It also resizes the images to 512 max on one side.

Sample images are not saved in this repo so that functionality will not work.

## Using CellPose.org an ImJoy plugin

This web application supports `imjoy-rpc`-- a protocol enables remote function calls beween ImJoy plugins.

This means you can use cellpose.org as an ImJoy plugin and passing images into it for segmentation.

Basically, you will be able to show an the cellpose window and call `segment` function in Python or Javascript:
``` python
cellpose = await api.createWindow(src="https://cellpose.org")
result = await cellpose.segment({"input": image, "outputs": "flow,mask,outline_plot,overlay_plot"})
```



