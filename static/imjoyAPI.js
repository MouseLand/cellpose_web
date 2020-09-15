'use strict';

const toBase64 = file => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result);
    reader.onerror = error => reject(error);
});

const ndarrayToBase64 = async (array) =>{
  let w, h, cc, depth;
  if(array._rdtype === 'uint16'){
    depth = 16;
  }
  else if(array._rdtype === 'uint8'){
    depth = 8;
  }
  else
    throw "dtype must be 'uint16' or 'uint8'";
  
  if(array._rshape.length === 2){
    cc = 1;
  }
  else if(array._rshape.length === 3){
    cc = array._rshape[2]
  }
  else{
    throw "invalid dimension number: " + array._rshape.length
  }
  w = array._rshape[1]
  h = array._rshape[0]
  // encode the array into a png file
  return await toBase64(new Blob([UPNG.encodeLL([array._rvalue], w, h, cc, 0, depth)]))
}

const base64ToNdArray = async base64 => {
  const dataUrl = "data:application/octet-binary;base64," + base64;
  const buffer = await fetch(dataUrl)
    .then(res => res.arrayBuffer())
  const imgObj = UPNG.decode(buffer)
  let dtype, channels;
  // gray
  if(imgObj.ctype === 0){
    channels = 1
  }
  // gray + alpha
  else if(imgObj.ctype === 4){
    channels = 2
  }
  // RGB + alpha
  else if(imgObj.ctype === 6){
    channels = 4
  }
  else{
    channels = 3
  }
  if(imgObj.depth === 16){
    dtype = 'uint16';
  }
  else if(imgObj.depth === 8){
    dtype = 'uint8';
  }
  else{
    throw "invalid depth: " + imgObj.depth
  }
  return {
    _rtype: "ndarray",
    _rshape: [imgObj.height, imgObj.width, channels],
    _rdtype: dtype,
    _rvalue: imgObj.data.slice(0, imgObj.height* imgObj.width* channels*(imgObj.depth/8)).buffer
  }
}

const plot_titles = {
  "img": "original image",
  "outline": "predicted outlines",
  "overlay": "predicted masks",
  "flow": "predicted cell pose"
}

async function runSegmentation(){
  const elm = document.getElementById("input-file");
  const result = await segment({input: elm.files[0], outputs: "mask,geojson,outline_plot,img_plot"})
  console.log(result)
}

document.addEventListener('DOMContentLoaded', function(){
  document.getElementById("loader").style.display = "none";
  document.getElementById("imjoy-display").style.display = "none";
  // check if it's inside an iframe
  // only activate for home page
  if(window.self !== window.top && location.pathname === '/'){
    loadImJoyRPC().then(async (imjoyRPC)=>{
        const api = await imjoyRPC.setupRPC({name: 'CellPose', description: 'a generalist algorithm for cellular segmentation'});
        async function setup(){
          await api.log('CellPose initialized.')
        }
        
        async function run(ctx){
          if(ctx.data && ctx.data.input){
            const results = await segment({input: ctx.data.input, outputs: "mask,flow,img_plot,overlay_plot,outline_plot,flow_plot"})
            console.log('CellPose segmentation results:', results)
          }
        }

        function saveConfig(config){
          // save config to localstorage
          window.localStorage.setItem("cellposeConfig", JSON.stringify(config));
        }

        function showConfig(){
          return new Promise((resolve, reject)=>{
            document.getElementById("main").style.display = "none";
            document.getElementById("imjoy-display").style.display = "block";
            document.getElementById("results").style.display = "none";
            document.getElementById("config").style.display = "block";
            try{
              const c = window.localStorage.getItem("cellposeConfig");
              if(c){
                // load config into the html form
                const savedConfig=JSON.parse(c);
                document.getElementById("net").value = savedConfig.net;
                document.getElementById("chan1").value = savedConfig.chan1;
                document.getElementById("chan2").value = savedConfig.chan2;
                document.getElementById("diam").value = savedConfig.diam;
                document.getElementById("invert").checked = savedConfig.invert;
                document.getElementById("keep-size").checked = savedConfig.keep_size;
              }
            }
            catch(e){
              console.error(e)
            }

            document.getElementById("save-config").onclick = ()=>{
              // obtain config from the html form
              const config = {
                net: document.getElementById("net").value,
                chan1: document.getElementById("chan1").value,
                chan2: document.getElementById("chan2").value,
                diam: document.getElementById("diam").value,
                invert: document.getElementById("invert").checked,
                keep_size: document.getElementById("keep-size").checked,
              }
              saveConfig(config);
              resolve(config)
              api.close();
            }
            api.on("close", ()=>{
              reject("closed")
            });
          })
        }

        async function segment(config){
          document.getElementById("loader").style.display = "block";
          document.getElementById("main").style.display = "none";
          document.getElementById("imjoy-display").style.display = "block";
          document.getElementById("results").style.display = "none";
          document.getElementById("config").style.display = "none";
          document.getElementById("results").innerHTML = "";
          let return_type = 'base64';
          try{
            let fileBase64 = config.input;
            if(config.input instanceof File){
              fileBase64 = await toBase64(config.input);
            }
            else if(config.input._rtype === 'ndarray'){
              fileBase64 = await ndarrayToBase64(config.input)
              // if the input type is ndarray, set the return type to ndarray
              return_type = 'ndarray';
            }
            else if(config.input.startsWith('http')){
              const blob = await fetch(config.input).then(r => r.blob());
              fileBase64 = await toBase64(blob);
            }

            if(fileBase64.startsWith('data:'))
              fileBase64 = fileBase64.split(';base64,')[1]

            let savedconfig = {}
            try{
              const c = window.localStorage.getItem("cellposeConfig");
              if(c) savedconfig=JSON.parse(c);
            }
            catch(e){
              console.error(e)
              savedconfig = {}
            }
            const formData = new FormData();
            formData.append('input', fileBase64);
            // formData.append('format', "png");
            formData.append('net', config.net || savedconfig.net || "cyto");
            formData.append('chan1', config.chan1 || savedconfig.chan1 || "0");
            formData.append('chan2', config.chan2 || savedconfig.chan2 || "0");
            formData.append('diam', config.diam || savedconfig.diam || "30");
            formData.append('invert', config.invert || savedconfig.invert || false);
            formData.append('keep_size', config.keep_size || savedconfig.keep_size || false);
            // possible outputs: geojson,mask,flow,img,outline_plot,overlay_plot,flow_plot,img_plot
            formData.append('outputs', config.outputs || "mask,outline_plot,overlay_plot,flow_plot,img_plot");
            
            const response = await fetch("/segment", {
              method: 'POST',
              mode: 'cors',
              body: formData
            })
            const result = await response.json()
            if(result.success){
              const resultsElm = document.getElementById("results");
              if(result.img_plot){
                const disp = document.createElement("div");
                disp.classList.add("column");
                disp.innerHTML = `
                <center>${plot_titles["img"]} </center>
                <img width="100%">
                `
                resultsElm.appendChild(disp)
                disp.children[1].src = "data:image/png;base64," + result.img_plot;
              }
              if(result.outline_plot){
                const disp = document.createElement("div");
                disp.classList.add("column");
                disp.innerHTML = `
                <center>${plot_titles["outline"]} </center>
                <img width="100%">
                `
                resultsElm.appendChild(disp)
                disp.children[1].src = "data:image/png;base64," + result.outline_plot;
              }
              
              if(result.overlay_plot){
                const disp = document.createElement("div");
                disp.classList.add("column");
                disp.innerHTML = `
                <center>${plot_titles["overlay"]} </center>
                <img width="100%">
                `
                resultsElm.appendChild(disp)
                disp.children[1].src = "data:image/png;base64," + result.overlay_plot;
              }
              if(result.flow_plot){
                const disp = document.createElement("div");
                disp.classList.add("column");
                disp.innerHTML = `
                <center>${plot_titles["flow"]} </center>
                <img width="100%">
                `
                resultsElm.appendChild(disp)
                disp.children[1].src = "data:image/png;base64," + result.flow_plot;
              }
              
              // convert base64 to ndarray
              if(return_type === 'ndarray'){
                if(result.mask){
                  result.mask = await base64ToNdArray(result.mask)
                }
                if(result.flow){
                  result.flow = await base64ToNdArray(result.flow)
                }
              }
              const info = document.createElement("P")
              info.innerHTML = `Done! Execution time: ${result.execution_time}; Timestamp: ${result.timestamp}`
              resultsElm.append(info);
              resultsElm.style.display = "block";
              return result
            }
            else
              throw new Error(result.error)
          }
          catch(e){
            document.getElementById("results").style.display = "none";
            throw e
          }
          finally{
            document.getElementById("loader").style.display = "none";
          }
        }
        // Importantly, you need to call `api.export(...)` in order to expose the api for your web application
        api.export({setup, run, segment, showConfig});
    })
  }
}, false);