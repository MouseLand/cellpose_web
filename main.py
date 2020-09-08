from flask import Flask, redirect, render_template, request, session, url_for, Response, send_file, send_from_directory
from werkzeug.datastructures import FileStorage
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
import urllib, os, io, datetime
import numpy as np
import mxnet as mx
import cv2
import matplotlib.pyplot as plt
import matplotlib
import base64
from cellpose import models, plot
import gc

matplotlib.rc('axes', edgecolor='w')
matplotlib.rc('xtick', color='w', labelsize=10)
matplotlib.rc('ytick', color='w', labelsize=10)

app = Flask(__name__)
dropzone = Dropzone(app)

app.config['SECRET_KEY'] = 'my-secret'
app.config["CACHE_TYPE"] = "null"

# Dropzone settings
app.config['DROPZONE_UPLOAD_MULTIPLE'] = False
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'image_plot'
app.config['DROPZONE_MAX_FILE_SIZE'] = 10
app.config['DROPZONE_MAX_FILES'] = 1

# Uploads settings
app.config['UPLOADED_PHOTOS_DEST'] = '/tmp' #os.getcwd() + '/uploads'
#'/tmp' #+ '/uploads'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

model = models.CellposeModel(device=mx.cpu(),
                       pretrained_model='static/models/cyto_0')

def url_to_image(file_url):
    resp = urllib.request.urlopen(file_url)
    img = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def image_resize(img, resize=512):
    ny,nx = img.shape[:2]
    if np.array(img.shape).max() > resize:
        if ny>nx:
            nx = int(nx/ny * resize)
            ny = resize
        else:
            ny = int(ny/nx * resize)
            nx = resize
        shape = (nx,ny)
        img = cv2.resize(img, shape)
    #iy = [np.maximum(0, ny//2 - 96), np.minimum(ny-1, ny//2+96)]
    #ix = [np.maximum(0, nx//2 - 96), np.minimum(nx-1, nx//2+96)]
    #img = img[iy[0]:iy[1], ix[0]:ix[1]]
    img = img.astype(np.uint8)
    return img

def sample_props():
    chan1 = [2,0,2,0,2,0,2,0,0,2,0,2,0,0,0,0,2,0,2,0,2,0,0]
    chan2 = []
    models = []
    images = []
    for k,c in enumerate(chan1):
        images.append('img%02d.png'%k)
        models.append(0)
        if c==2:
            chan2.append(3)
        else:
            chan2.append(0)
    return images, models, chan1, chan2

def remove_temp():
    for f in os.listdir('/tmp'):
        fpath = os.path.join('/tmp', f)
        try:
            os.remove(fpath)
        except:
            print('directory %s not removed'%fpath)

def img_to_html(img, outpix=None, axis_on=False):
    figsize = (6,6)
    if img.shape[0]>img.shape[1]:
        figsize = (6*img.shape[1]/img.shape[0], 6)
    else:
        figsize = (6, 6*img.shape[0]/img.shape[1])
    fig = plt.figure(figsize=figsize, facecolor='k')
    ax = fig.add_axes([0.08,0.08,.84,.84])
    ax.set_xlim([0,img.shape[1]])
    ax.set_ylim([0,img.shape[0]])
    ax.imshow(img[::-1], origin='upper')
    if outpix is not None:
        for o in outpix:
            ax.plot(o[:,0], img.shape[0]-o[:,1], color=[1,0,0], lw=1)
    if not axis_on:
        ax.axis('off')
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
    bytes_image.seek(0)
    img_html = base64.b64encode(bytes_image.getvalue()).decode()
    del bytes_image
    fig.clf()
    plt.close(fig)
    return img_html

@app.route('/docs')
def docs():
    #return redirect(url_for('static', filename='docs/index.html'))
    return redirect('https://cellpose.readthedocs.io')

@app.route('/image')
def image_plot():
    if "file_url" not in session or session['file_url'] == []:
        return redirect(url_for('index'))
    try:
        img = url_to_image(session['file_url'])
        img = image_resize(img)
        img_html = img_to_html(img, axis_on=True)
        del img
        gc.collect()
        return render_template('image.html', filename='user',
                            model=0, chan1=0, chan2=0, result=img_html)
    except:
        return redirect(url_for('index'))

@app.route('/image/<filename>/<model>/<chan1>/<chan2>')
def image_plot_demo(filename, model, chan1, chan2):
    img = cv2.imread('static/images/' + filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_html = img_to_html(img, axis_on=True)
    del img
    gc.collect()
    return render_template('image.html', filename=filename,
                           model=int(model), chan1=int(chan1),
                           chan2=int(chan2), result=img_html)

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'file_url' in session and len(session['file_url']) > 0:
        imgpath = '/tmp/' + session['file_url'].split("/")[-1]
        try:
            os.remove(imgpath)
        except:
            session['file_url'] = []
    #if 'model_type' not in session:
    #    session['model_type'] = 'cyto'

    session['file_url'] = []
    file_url = []
    # handle image upload from Dropszone
    if request.method == 'POST':
        session['time'] = datetime.datetime.now()
        session['file_url'] = []
        file_url = []
        file_obj = request.files
        k=0
        for f in file_obj:
            if k==0:
                file = request.files.get(f)
                time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S.%f")
                filestring = time + os.path.splitext(file.filename)[0]
                # save the file to our photos folder
                filename = photos.save(file, name=filestring+'.')
                # append image urls
                file_url.append(photos.url(filename))
                session['filestring'] = filestring
            k+=1
        session['file_url'] = file_url[-1]
        return "uploading..."
    else:
        result = None
        images, mdls, chan1, chan2 = sample_props()
        gc.collect()
        return render_template('index.html', result=result,
                               img=images, model=mdls, chan1=chan1, chan2=chan2)

@app.route('/results/<filename>', methods=['POST'])
def results(filename):
    if request.method == 'POST':
        result = request.form
        val = []
        if filename=='user':
            for key, value in result.items():
                val.append(value)
            img = url_to_image(session['file_url'])
            if val[0]=='cyto':
                diam_mean = 27
            else:
                diam_mean = 15
            try:
                if float(val[-1])==30:
                    rsz = 1.0
                else:
                    rsz = diam_mean/(float(val[-1])*(np.pi**0.5/2))
            except:
                rsz = 1.0
            rsz = np.minimum(2., rsz)
            #isize = int(rsz * min(288, min(img.shape[0], img.shape[1])))
            img = image_resize(img)
            if img.ndim<3:
                img = img[:,:,np.newaxis]
            model.net.load_parameters('static/models/%s_0'%val[0])
            model.net.collect_params().setattr('grad_req', 'null')
            channels = [int(val[1]), int(val[2])]
            if val[0]!='cyto':
                channels[1] = 0
            masks, flows, _ = model.eval([img], rescale=[rsz], channels=channels, tile=False, jit=True)
            masks, flows = masks[0], flows[0][0]
            if channels[1]==0:
                if channels[0]==0:
                    img = np.tile(np.uint8(np.float32(img).mean(axis=-1))[:,:,np.newaxis], (1,1,3))
                else:
                    for i in range(img.shape[-1]):
                        if i!=channels[0]-1:
                            img[:,:,i] = 0
            #masks, flows = np.zeros_like(img[:,:,0]), np.zeros_like(img[:,:,0])
            outpix = plot.plot_outlines(masks)
            overlay = plot.mask_overlay(img, masks)
            overlay_outlines_html = img_to_html(img, outpix=outpix)
            gc.collect()
            plt.imsave('testfile.png', masks)
            with open('testfile.png', 'rb') as f:
                stream = io.BytesIO(f.read())
            del masks, outpix

            file = FileStorage(stream=stream, filename='masks1.png')
            session['filestring_masks'] = session['filestring'] + '_masks.png'
            filename = photos.save(file, name=session['filestring_masks'])
            session['file_url_masks'] = photos.url(filename)
            download_string = 'Download masks as PNG'
        else:
            _, mdls, chan1, chan2 = sample_props()
            val = []
            val.append('cyto')
            val.append(chan1[int(filename[-6:-4])])
            val.append(chan2[int(filename[-6:-4])])
            val.append('30')
            img = cv2.imread('static/images/' + filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fileroot = os.path.splitext(filename)[0]
            overlay = plt.imread('static/segs/' + fileroot + '_overlay.jpg')
            outlines = plt.imread('static/segs/' + fileroot + '_outlines.jpg')
            flows = plt.imread('static/segs/' + fileroot + '_flows.jpg')
            overlay_outlines_html = img_to_html(outlines)
            masks = flows
            gc.collect()
            download_string = ''

        overlay_masks_html = img_to_html(overlay)
        flow_html = img_to_html(flows)
        gc.collect()
        img_html = img_to_html(img)
        gc.collect()
        del img, overlay, flows
        gc.collect()

        return render_template('results.html', #outlines=img_html, masks=img_html, flow=img_html, image=img_html)
                                outlines=overlay_outlines_html,
                                masks=overlay_masks_html,
                                flow=flow_html,
                                image=img_html,
                                download_string=download_string)
        #return render_template('results.html', file_url='/tmp/%s_masks.png'%session['time'])


@app.route("/models/<filename>")
def download_models(filename):
    return send_file('static/models/' + filename, as_attachment=True)

@app.route("/download_masks")
def download_masks():
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'],
                               session['filestring_masks'], as_attachment=True)

# retrieve file from 'static/images' directory
@app.route("/tmp/<filename>")
def send_image(filename):
    return send_from_directory(filename='/tmp/%s'%filename)

#@app.route('', defaults={'static': True})
#def doc(''):
#    path = join(dir,filename)
#    return app.send_static_file(path)

#@app.route("/documentation")
#def documentation():
#    return render_template('static/source/_build/html/index.html')

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
