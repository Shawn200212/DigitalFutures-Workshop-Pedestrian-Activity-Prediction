# Offline GraphSAGE Result Explorer

**English** | [中文](README.zh-CN.md)

This folder contains the offline visualization website for the Shanghai street-level pedestrian
activity GraphSAGE model.

## Open it

Double-click `打开模型可视化网页.bat` or open `index.html` directly.

## Keep the files together

- `index.html`
- `app.js`
- `styles.css`
- `data/app_data.js`
- `打开模型可视化网页.bat`
- `模型可视化网页使用与视频制作文档.md`

## Features

- 2D top-down spatial prediction map
- immersive 3D result exploration
- prediction, ground-truth, residual, absolute-error, hotspot-score, and hotspot-class layers
- split and tile filters
- node hover details
- variable-group and individual-variable influence views
- PNG export
- camera presets for video production

## Portable offline use

Copy the complete folder to another computer, or extract the supplied ZIP and double-click the
launcher. Python, Conda, Node, and model dependencies are not required.
