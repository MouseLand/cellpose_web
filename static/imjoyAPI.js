'use strict';

const toBase64 = file => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result);
    reader.onerror = error => reject(error);
});
const plot_titles = {
  "img": "original image",
  "outline": "predicted outlines",
  "overlay": "predicted masks",
  "flow": "predicted cell pose"
}
async function segment(config){
  document.getElementById("loader").style.display = "block";
  document.getElementById("main").style.display = "none";
  document.getElementById("result-display").style.display = "block";
  document.getElementById("results").style.display = "none";
  document.getElementById("results").innerHTML = "";
  try{
    let fileBase64 = config.input;
    if(config.input instanceof File){
      fileBase64 = await toBase64(config.input);
    }
    if(fileBase64.startsWith('data:'))
      fileBase64 = fileBase64.split(';base64,')[1]
    const formData = new FormData();
    formData.append('input', fileBase64);
    // formData.append('format', "png");
    formData.append('net', config.net || "cyto");
    formData.append('chan1', config.chan1 || "0");
    formData.append('chan2', config.chan2 || "0");
    formData.append('diam', config.diam || "30");
    // possible outputs: geojson,mask,flow,img,outline_plot,overlay_plot,flow_plot,img_plot
    formData.append('outputs', config.outputs || "outline_plot,overlay_plot,flow_plot,img_plot");
    
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
async function runSegmentation(){
  const elm = document.getElementById("input-file");
  const result = await segment({input: elm.files[0], outputs: "mask,geojson,outline_plot,img_plot"})
  console.log(result)
}

document.addEventListener('DOMContentLoaded', function(){
  const display = document.createElement('DIV')
  display.id = "result-display"
  display.innerHTML = '<img id="loader" src="static/images/loader.gif" style="width:300px;">' +
    '<div id="results" class="row"></div>';
  document.body.appendChild(display);
  document.getElementById("loader").style.display = "none";
  document.getElementById("result-display").style.display = "none";
  // check if it's inside an iframe
  // only activate for home page
  if(window.self !== window.top && location.pathname === '/'){
    loadImJoyRPC().then(async (imjoyRPC)=>{
        const api = await imjoyRPC.setupRPC({name: 'CellPose', description: 'a generalist algorithm for cellular segmentation'});
        function setup(){
            api.log('CellPose initialized.')
        }
        
        function run(){

        }
        // Importantly, you need to call `api.export(...)` in order to expose the api for your web application
        api.export({setup, run, segment});
    })
  }
}, false);