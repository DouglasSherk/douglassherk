---
layout: post
title: ROS, Robot Operating System
date: 2013-03-17 00:00:00 -0000
categories: Coding
---

## Preface

This article assumes that you already know what ROS is and the very basics about it. If you don't know, check it out at <http://ros.org>

There are already many guides that exist for ROS, explaining how to install and use it, but they're pretty bad, so I want to try to make a good one that has a better signal-to-noise ratio. Most of the guides are either extremely unclear or stop before explaining any functionality, API, or building projects.

This article is fairly opinionated (at least at the beginning), based on what I think are good and bad design principles. I may also overly boil down concepts to make them more clear, but not entirely correct if you really get into ROS.

## Disclaimer

I haven't used ROS much. The only thing I've used it for is [myHelperBot](https://anselm.ca/myhelperbot) ([source code](https://github.com/DouglasSherk/myHelperBot)), which if you go looking, you'll notice actually has no ROS code. That's because we had to rip it out since it was bad and wasn't working well. I've heard from several people who took the [ME 597](http://www.me.uwaterloo.ca/~me597/index.html) course at UWaterloo that they spent most of their time just trying to get ROS working. I don't know, because I didn't take this course, but stumbled onto ROS in a different way.

Part of the reason I'm writing this is because ROS is difficult to start with and get into, and I want to ease the pain for other people. I think despite being a fundamentally good and useful idea, the execution of it is so bad that it is difficult for me to recommend it. For this reason, I strongly recommend not using ROS if there are other potential solutions available to you. For example, for myHelperBot, we threw away all of our code and started from scratch in .NET/C#. It turned out that it ran much better, had far more documentation to get up and running with, and was much easier to set up. With that out of the way, let's talk about ROS.

## Advantages of ROS

- Installed via apt-get (ideally) and thus it's pretty easy to get the basics set up (huge footnote here).*
- Relatively good separation of unrelated code, making ownership of responsibilities fairly clear.
- Great, portable, message passing IPC. The handling of message passing to Arduino is superb and very well done.
- Tons of example code and hack projects.
- Supports multiple languages running side-by-side without having to embed them in each other (C++, Python, and Lisp). Very handy.
- You can write your code to be highly distributed and even network it, while running the core on one master controller.

\* while this sounds great and is indeed useful, I hear of people having to build from source quite often, especially with newer builds

## Disadvantages of ROS

- Actually getting started is very convoluted and unnecessarily difficult.
- The documentation needs work, and community support is lacklustre. While it exists, most problems I had took quite a while of Googling to actually find a useful answer. Most of the answers you need are out there, but are drowned out by relatively useless information.
- Very complex and clunky for small projects, with tons of features that I'm guessing few people use.
- Error messages while installing and trying to run code are fairly useless.
- Packages (basically synonymous with a package in any package manager) take far too long to rebuild.
- Several executables have to be run at the same time to do anything useful. For example, to have a Kinect <=> Computer <=> Arduino interface, you need to run: ROS core, ROS serial Arduino helper, ROS Kinect library/helper, ROS Kinect custom code, ROS Arduino custom code.
- Given libraries and base code don't recover well from errors. For example, the ROS serial Arduino helper crashes if you unplug the Arduino.
- The concurrency model is fairly unclear, though powerful and quick to get into if you just need to get something working quickly.
- Example code and hack projects may or may not work. They often link to dead pages, are based on old API's with no backwards compatibility, or depend on hardware that you may or may not have.
- If you're using ROS for the Kinect functionality, don't even bother. The Microsoft Kinect SDK is so much better that it's not even worth considering OpenNI on ROS.

## Installation

*credit in this section goes mostly to [the ROS website](http://www.ros.org/wiki/electric/Installation/Ubuntu); I'm largely just copying and pasting*

With all that out of the way, let's get started with actually installing, explaining, and then using ROS.

I highly recommend using Ubuntu 11.10, regardless of the fact that it's old. I used Ubuntu 12.10 and a lot of stuff was completely broken. They may or may not have fixed it since then, but the only version I can vouch for is 11.10. I'm guessing 11.04 might be fine too, but wouldn't bet on it. I'm going to write the rest of this assuming that you're running Ubuntu 11.10, for the sake of clarity. We're also going to be setting up ROS Electric, which is one of the distros of ROS that I can confirm works well.

Add the ROS code to your source repos:

```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu oneiric main" > /etc/apt/sources.list.d/ros-latest.list'
```

Set up keys to authenticate the source repos with:

```bash
wget http://packages.ros.org/ros.key -O - | sudo apt-key add -
```

Update your repos:

```bash
sudo apt-get update
```

Install ROS:

```bash
sudo apt-get install ros-electric-desktop-full
```

Setup environment:

```bash
echo "source /opt/ros/electric/setup.bash" >> ~/.bashrc
. ~/.bashrc
```

Install the rosinstall package, which lets you manage your workspace.

```bash
sudo apt-get install python-rosinstall
```

You now have everything installed, but you can't do anything useful with it yet.

To verify that everything is working (or at least a good litmus test), **close your current terminal and open a new one**, then type:

```bash
roscore
```

## Important Concepts

I think it's best to get the installation out of the way so you can start focusing entirely on understanding things. If you haven't installed it yet, you should go back to the previous step and do that.

There are some important concepts to understand about ROS before getting started.

- *Publisher*: A block of code which pushes (publishes) messages to other nodes (subscribers). Nodes will be explained later.
- *Subscriber*: A block of code which receives (subscribes to) messages from other nodes (publishers).
- *Workspace*: A location where all of your custom code and resources are stored.
- *Package*: An independent set of code that can interface with other packages. It is designed to be redistributable, though you don't have to be careful about this, unless you want to redistribute your packages. Note that packages can refer both to "stock" code, as well as your custom code. Your custom packages reside within your workspace, while stock code resides in the ROS installation folder.
- *Node*: Possibly the most useful part of ROS, nodes are an independent unit of execution, designed to operate alongside other nodes. Nodes are a subset of packages. That is, usually you have one or more nodes within a package. For example, a ROS package that polls a sensor for data and adjusts a motor based on the sensor data may be divided into:
  - Sensor node, which polls sensor data and publishes a message containing this data.
  - Logic node, which subscribes to sensor data messages and publishes a motor power message containing a corresponding motor setting.
  - Motor node, which subscribes to motor power messages and sets the motor power.

  Of note here is that nodes can run on different hardware. For example, the sensor and motor nodes could run on two different Arduinos, and the logic node could run on a desktop, and ROS will figure this out.
- *Launcher*: A tool that lets you launch several nodes at the same time, for the purposes of launching all the nodes a package needs, rather than forcing the user to figure out what to do with a package.

## What's Next

Now that you have everything installed, the next steps will be:

1. Create a workspace.
2. Create a package.
3. Create a node.
4. Build your project (the package).
5. Run your project. *After this point you can stop if you want as not everyone wants the stuff past here.*
6. Create another node, and edit the first one (message passing example).
7. Create a launcher.
8. Interface with an Arduino.
9. Learn some useful commands.

## Create a Workspace

Your workspace will store everything you're working on. It can be messy (have code you're not using, etc.) and it really doesn't matter.

To create:

```bash
rosws init ~/workspace /opt/ros/electric
```

The first path is, of course, wherever you want it to be. Note that it can be basically anywhere, but for the purposes of this article, I'm going to assume that you put it in *~/workspace*. The second path tells ROS where to look for libraries and stock code from when dealing with this workspace.

Workspaces are otherwise not very notable. I don't recommend moving them around.

## Create a Package

There's a concept related to packages called stacks, which doesn't seem very useful to me. You can read about it on the ROS site but I never needed them. They're a way of organizing sets of packages.

To create a package, **first cd into your workspace**, then type:

```bash
roscreate-pkg packagename
```

To make sure it worked, type:

```bash
rospack find packagename
```

If it didn't work, that probably means that your package is not in the ROS_PACKAGE_PATH. This happened to me. I fixed it by just adding the workspace manually to ROS_PACKAGE_PATH in my .bashrc:

```bash
ROS_PACKAGE_PATH=/home/user/workspace:$ROS_PACKAGE_PATH
```

After adding that, type:

```bash
source ~/.bashrc
```

Then you should be good to go.

We're not going to put any actual code in here, yet.

## Create a Node

The node is where our actual code will go. We're going to create one in C++ for now, since for almost any project you're going to need some C++ anyways.

Navigate to *~/workspace/packagename/src/* and create a file called *mynode.cpp*. Then copy and paste the following into it:

```text
[Embedded GitHub gist omitted: https://gist.github.com/DouglasSherk/ad6fe8587cc7eb7ca437]
```

This is basically the bare minimum that could possibly be useful in any way, and is a decent place to start for any node. This should be fairly self-explanatory.

Open *CMakeLists.txt* and add to the very bottom:

```cmake
rosbuild_add_executable(mynode src/mynode.cpp)
```

Additional nodes are added the same way. We'll cover that more in detail in another section.

## Build your Package

We're ready to actually build your package. While in *~/workspace/packagename/*, type:

```bash
rosmake
```

When complete, your package will be ready to run.

## Running your Project

To run this node, first you must be running the ROS core. I recommend creating two terminal tabs for this (or you could just use screen). Type:

```bash
roscore
```

You should get a welcome message. Next, type:

```bash
rosrun packagename mynode
```

Your code is now running. You can stop here or keep going. While the code is running, this is a pretty useless example, and doesn't show anything about ROS other than the main loop. In the next sections, we'll go over useful examples.

## Create Another Node, Publisher/Subscriber

A far more useful example than the previous one is a publisher/subscriber model. To do this, we're going to edit the first node to be a publisher, and the second one to be a subscriber.

First, rename *mynode.cpp* to *publisher.cpp* and fix it in *CMakeLists.txt*.

Next, delete everything in it, and paste this into it instead.

#### publisher.cpp

```text
[Embedded GitHub gist omitted: https://gist.github.com/DouglasSherk/1652daa0819b8baef274]
```

Next, create a file called *subscriber.cpp* in the same folder, and paste this into it:

#### subscriber.cpp

```text
[Embedded GitHub gist omitted: https://gist.github.com/DouglasSherk/82c0276432d4f1e332fd]
```

Add *subscriber.cpp* to your *CMakeLists.txt*. At the end, it should now look like:

```cmake
rosbuild_add_executable(publisher src/publisher.cpp)
rosbuild_add_executable(publisher src/publisher.cpp)
```

You can find good and more detailed instructions and explanations on the ROS website [here](http://www.ros.org/wiki/ROS/Tutorials/WritingPublisherSubscriber(c%2B%2B)).

You can either run these two now, or wait until we build a launcher for them, which makes it easier to run them together. If you want to run them now, type in one terminal:

```bash
rosrun packagename publisher
```

Then type in another:

```bash
rosrun packagename subscriber
```

In the publisher window, you should see something like:

```text
[INFO] [WallTime: 1314931831.774057] hello world 1314931831.77
[INFO] [WallTime: 1314931832.775497] hello world 1314931832.77
[INFO] [WallTime: 1314931833.778937] hello world 1314931833.78
[INFO] [WallTime: 1314931834.782059] hello world 1314931834.78
[INFO] [WallTime: 1314931835.784853] hello world 1314931835.78
[INFO] [WallTime: 1314931836.788106] hello world 1314931836.79
```

And in the subscriber window, you should see something like:

```text
[INFO] [WallTime: 1314931969.258941] /subscriber_17657_1314931968795I heard hello world 1314931969.26
[INFO] [WallTime: 1314931970.262246] /subscriber_17657_1314931968795I heard hello world 1314931970.26
[INFO] [WallTime: 1314931971.266348] /subscriber_17657_1314931968795I heard hello world 1314931971.26
[INFO] [WallTime: 1314931972.270429] /subscriber_17657_1314931968795I heard hello world 1314931972.27
[INFO] [WallTime: 1314931973.274382] /subscriber_17657_1314931968795I heard hello world 1314931973.27
[INFO] [WallTime: 1314931974.277694] /subscriber_17657_1314931968795I heard hello world 1314931974.28
[INFO] [WallTime: 1314931975.283708] /subscriber_17657_1314931968795I heard hello world 1314931975.28
```

## Create a Launcher

You'll notice that the main weakness of this setup so far is that you have to have 3 terminal windows open, one for each of: the core, the subscriber, the publisher. Fortunately, we have an easy way to launch multiple nodes at once, called a launcher. Create a file called *packagename.launch* inside the *~/workspace/packagename/* folder, and paste this into it:

#### packagename.launch

```text
[Embedded GitHub gist omitted: https://gist.github.com/DouglasSherk/eaa83054d9120e655cb9]
```

Now to launch both of the nodes, you can type:

```bash
roslaunch packagename
```

Note that you still have to run roscore separately.

## Interface with Arduino

One of the most useful things about ROS is its ability to relatively seamlessly interface with Arduino. This implementation even scales well, supporting tons of message passing that as a non-expert at serial communication I have not been able to replicate.

To get started, we have to install the ROS serial package:

```bash
sudo apt-get install ros-electric-rosserial
```

With this installed, we have to copy ros_lib (a set of Arduino libraries and examples) to the Arduino sketchbook folder. The sketchbook folder is the location that Arduino saves to by default. If you're not sure where it is, you can also find it in the Arduino IDE's settings. You can also change it there. To copy ros_lib over, type the following:

```bash
roscd rosserial_arduino/libraries
mkdir -p sketchbook/libraries
cp -r ros_lib sketchbook/libraries/ros_lib
```

Now restart Arduino, then go to File>Examples and you should see a ros_lib folder with some examples in it.

Next, we're going to create a new Arduino sketch and call it *publisher.pde*. Paste the following code into it:

#### publisher.pde

```text
[Embedded GitHub gist omitted: https://gist.github.com/DouglasSherk/66c99d8bb35c270864aa]
```

Compile and upload this to your Arduino. Once complete, make sure that roscore is running, and then run the ROS serial helper:

```bash
rosrun rosserial_python serial_node.py /dev/ttyUSB0
```

Note that the port may not be named */dev/ttyUSB0*. You can find its name in the Arduino IDE's "Port" menu.

Also note that this node will segfault whenever you unplug the Arduino. You will have to rerun it when this happens. Also, to upload code to your Arduino, you have to first shut this node down, then upload the code, then rerun the serial node.

Finally, run the subscriber on your main computer (don't run the launcher):

```bash
rosrun packagename subscriber
```

Your Arduino should now be communicating with your computer. If not, make sure the following are connected and running:

- roscore
- ROS serial helper
- Subscriber node (on the computer)
- Arduino (with publisher code on it)

You can find more tutorials and explanations on the ROS website [here](http://www.ros.org/wiki/rosserial_arduino/Tutorials).

## Learn some Useful Commands

Some useful commands that didn't make it into any of the instructions:

- `rostopic echo chatter` - echoes published messages, so you can figure out if problems are on the subscriber or publisher side (`chatter` is the topic to echo).
- `roswtf` - displays errors and warnings about anything running attached to ROS.
- `roscd` - cd's into a package directory, whether it's with the ROS installation or in your workspace.
- Find a ROS cheat sheet [here](http://www.ros.org/wiki/Documentation?action=AttachFile&do=get&target=ROScheatsheet.pdf).

## Conclusion

That's it. I also have some experience getting the Kinect stuff running, but I wouldn't recommend using it at all. If enough people want me to write a tutorial on that, maybe I will.
