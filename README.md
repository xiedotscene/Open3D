# Open3D: A Modern Library for 3D Data Processing

[![Build Status](https://travis-ci.org/IntelVCL/Open3D.svg?branch=master)](https://travis-ci.org/IntelVCL/Open3D)
[![Build status](https://ci.appveyor.com/api/projects/status/sau3yewsyxaxpkqe?svg=true)](https://ci.appveyor.com/project/syncle/open3d)

## About this project

Open3D is an open-source library that supports rapid development of software that deals with 3D data. The Open3D frontend exposes a set of carefully selected data structures and algorithms in both C++ and Python. The backend is highly optimized and is set up for parallelization. We welcome contributions from the open-source community.

Please cite our work if you use Open3D.
```
@article{Zhou2018,
	author    = {Qian-Yi Zhou and Jaesik Park and Vladlen Koltun},
	title     = {{Open3D}: {A} Modern Library for {3D} Data Processing},
	journal   = {arXiv:1801.09847},
	year      = {2018},
}
```

## Core features

* Basic 3D data structures
* Basic 3D data processing algorithms
* Scene reconstruction
* Surface alignment
* 3D visualization
* Python binding

## Supported compilers

* GCC 4.8 and later on Linux
* XCode 8.0 and later on OS X
* Visual Studio 2015 and later on Windows

## Resources

* Website: [www.open3d.org](http://www.open3d.org)
* Code: [github.com/IntelVCL/Open3D](https://github.com/IntelVCL/Open3D)
* Document: [www.open3d.org/docs](http://www.open3d.org/docs)
* License: [The MIT license](https://opensource.org/licenses/MIT)

## Remark
* Add `Test` and `Helper` directories under `src/Python` to avoid compile error
