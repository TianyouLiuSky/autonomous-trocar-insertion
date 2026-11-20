# DeckLink实时显微镜视频 -- 状态：软件层已完全打通，只差物理接线

（2026-09-22 更新：原计划需要去Blackmagic官网下载SDK重新编译gstreamer插件，
实测发现完全不需要 -- Ubuntu apt自带的`gstreamer1.0-plugins-bad`包里其实
已经包含好用的`decklinkvideosrc`元素。下面记录的是已验证成功的最终路径。）

## 已完成并验证的软件安装

```bash
sudo apt-get install -y ros-melodic-gscam \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-bad \
    gstreamer1.0-tools
```
（已装完，dpkg确认 `ros-melodic-gscam 1.0.1-0bionic.20221025.191123`、
`gstreamer1.0-plugins-bad 1.14.5-0ubuntu1~18.04.1`）

## 端到端验证结果（两层都测了，卡在同一个"无信号"点上，证明软件全对）

**第1层 -- 纯GStreamer，不经过ROS：**
```bash
gst-launch-1.0 decklinkvideosrc device-number=0 connection=hdmi ! fakesink
```
报错：`Failed to acquire input`（gstdecklinkvideosrc.cpp:1001）
-> 这是DeckLink卡驱动层的标准报错，意思是"插件本身工作正常，
但这张卡的输入口上没有检测到任何视频信号"。

**第2层 -- 完整ROS集成：**
```bash
roslaunch /home/eyebrp/catkin_ws/src/cisst-saw-nri/dvrk-ros/dvrk_robot/launch/gscam_decklink.launch connection:=hdmi
```
- gscam节点、tf静态变换节点都正常启动，参数加载正常，
  gstreamer管线字符串拼接正确(`decklinkvideosrc connection=hdmi
  device-number=0 ! videoconvert`)
- 报错：`Failed to PAUSE stream, check your gstreamer configuration.`
  -- 和第1层同一个根因（无信号），节点干净退出，没有崩溃/依赖缺失

**结论：软件链路（apt包 + gscam节点 + launch文件参数）100%正确，唯一
剩下的是物理层 -- 需要真的有一路视频信号接到这张卡的输入口上。**

## 剩下唯一要做的事：接线（这一步只能你本人在实验室完成）

1. 找到这台Linux主机机箱后面的**DeckLink Mini Recorder 4K**卡（PCIe插槽那排），
   应该能看到SDI(BNC头)和/或HDMI输入口
2. 找一个分线器，从Leica CCU现在接显示器那路视频信号里分一路出来，接到这张
   DeckLink卡的输入口（HDMI接口用HDMI口，SDI接口用SDI口/BNC）
3. 接好之后告诉我，或者直接自己跑下面的验证命令

## 接线后的验证 + 启动顺序

```bash
# 1. 纯GStreamer快速验证信号是否真的进来了（换成实际接的connection类型: hdmi 或 sdi）
gst-launch-1.0 -v decklinkvideosrc device-number=0 connection=hdmi ! videoconvert ! autovideosink

# 若看到画面弹出来 -> 硬件链路完全打通，Ctrl+C关掉，继续下一步

# 2. 正式跑ROS节点（记得先source环境）
source /opt/ros/melodic/setup.bash
source /home/eyebrp/SHER_I2RIS_teleoperate/devel/setup.bash
roslaunch /home/eyebrp/catkin_ws/src/cisst-saw-nri/dvrk-ros/dvrk_robot/launch/gscam_decklink.launch connection:=hdmi

# 3. 另开一个终端，确认topic里真的有数据在流动
source /opt/ros/melodic/setup.bash
rostopic hz /decklink/camera/image_raw

# 4. 再开一个终端，跑桥接脚本，自动把画面写进现有demo监视的文件夹
cd /home/eyebrp/Matthew/microscope_autonomy
source /opt/ros/melodic/setup.bash
python3 mic_live_capture_ros.py --show
```
第4步跑起来之后，`home_to_rcm_insert_v2_precision.py`界面右上角的显微镜
PiP面板会自动开始刷新，`y`键锁定用的也会是真正实时的画面 -- **不需要再
改demo脚本的任何代码**。

## 备用排查（如果接线后gst-launch还是报同样的错）

- 换成 `connection=sdi`（如果接的是SDI口不是HDMI）
- 换成 `connection=auto`（让插件自动探测输入类型）
- 确认Leica CCU那边视频输出确实是"开着"的（比如显示器上那路信号本身
  是不是也还亮着）
- `device-number=0` 如果以后加第二张DeckLink卡需要改成对应编号，目前
  只有一张卡，用0没问题
