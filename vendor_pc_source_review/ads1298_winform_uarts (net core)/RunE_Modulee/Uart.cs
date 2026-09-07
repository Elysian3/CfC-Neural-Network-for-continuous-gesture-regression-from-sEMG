using RunE_Moudule;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO.Ports;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Xml.Linq;
using static System.Windows.Forms.VisualStyles.VisualStyleElement;

namespace RunE_Module
{
    public partial class Uart : UserControl
    {
        /// <summary>
        /// 串口对象
        /// </summary>
        private readonly SerialPort serialPort = new SerialPort();

        /// <summary>
        /// 数据缓存及解析
        /// </summary>
        public ADS1298_Data AdsData  = new ADS1298_Data();

        /// <summary>
        /// 串口打开状态
        /// </summary>
        public bool UartOpenState = false;

        /// <summary>
        /// 记录数据包接收时间
        /// </summary>
        private DateTime ConnectTime = new DateTime(2015, 12, 31);

        /// <summary>
        /// 采集状态
        /// </summary>
        public bool SampleState = false;

        public Uart()
        {
            InitializeComponent();
            UpdateConnState(false);
        }

        private void Uart_Load(object sender, EventArgs e)
        {
            CbSignalType.SelectedIndex = 0;
            CbRate.SelectedIndex = 0;
            CbRange.SelectedIndex = 6;

            //检查是否含有串口
            string[] portNames = SerialPort.GetPortNames();
            if (portNames != null)
            {
                //更新串口列表
                CbSerial.Items.AddRange(portNames);

                //使用串口接收事件函数，接收数据
                serialPort.DataReceived += new SerialDataReceivedEventHandler(SerialPort_DataReceived);

                //准备就绪              
                serialPort.DtrEnable = false;
                serialPort.RtsEnable = false;
                serialPort.ReadTimeout = 1000;  //设置数据读取超时为1秒
                serialPort.Close();             //先关闭串口
            }

            Task task = new Task(async () =>
            {
                while (true)
                {
                    await Task.Delay(1000);

                    if (!serialPort.IsOpen)
                    {
                        UpdateConnState(false);
                        continue;
                    }


                    // 如果不在采集状态，则定时向下位机发送，连接状态更新指令
                    if (!SampleState)
                    {
                        // 发送连接状态更新指令：保持连接
                        SendCmd(ADS1298_Cmd.ConnStatusUpdate(2));
                    }

                    try
                    {
                        this.Invoke(new Action(() => {
                             TimeSpan timeSpan =  DateTime.Now - ConnectTime;
                             UpdateConnState(timeSpan.TotalSeconds < 10);
                        }));
                    }
                    catch { }
                }
            });
            task.Start();
        }

        /// <summary>
        /// 接收串口数据事件
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void SerialPort_DataReceived(object sender, SerialDataReceivedEventArgs e)
        {
            if (serialPort.IsOpen)
            {
                int n = serialPort.BytesToRead;
                byte[] receivedData = new byte[n];        //创建接收字节数组
                serialPort.Read(receivedData, 0, receivedData.Length);             //读取数据 

                //for (int i = 0; i < receivedData.Length; i++)
                //{
                //    string a = receivedData[i].ToString("X2");
                //    Console.Write("Ox" + a.ToString() + "  ");
                //}

                //string str = Encoding.Default.GetString(receivedData);
                //Console.WriteLine(str);
                //Tra.WriteLine("1234===" + receivedData.Length);

                AdsData.AddData(receivedData);
                ConnectTime = DateTime.Now;
            }
            else
            {
                //MessageBox.Show("请打开某个串口", "错误提示");
            }
        }

        /// <summary>
        /// 更新连接状态
        /// </summary>
        private void UpdateConnState(bool connState)
        {
            if (connState)
            {
                lbConnState.Text = "已连接";
                lbConnState.ForeColor = Color.Green;
                pictureBox1.Image = CreateBatteryIcon((int)AdsData.BatteryLevel);
                toolTip1.SetToolTip(pictureBox1, $"电量:{(int)AdsData.BatteryLevel}%");
            }
            else
            {
                lbConnState.Text = "未连接";
                lbConnState.ForeColor = Color.Red;
                pictureBox1.Image = CreateBatteryIcon((int)0);
                toolTip1.SetToolTip(pictureBox1, "电量:0%");
            }
        }

        /// <summary>
        /// 打开串口
        /// </summary>
        /// <returns></returns>
        public bool OpenSerial()
        {
            UartOpenState = false;
            try
            {
                // ESP32与C#的串口流控导致的重启问题解析
                // https://elmagnifico.tech/2022/09/12/FlowControl-ESP32/
                if (!serialPort.IsOpen && CbSerial.Items.Count > 0 && CbSerial.SelectedItem != null)
                {
                    serialPort.PortName = CbSerial.SelectedItem.ToString();   //串口号
                    //serialPort.BaudRate = 256000;  //波特率
                    serialPort.BaudRate = 1000000;  //波特率
                    serialPort.DataBits = 8;       //数据位
                    serialPort.StopBits = StopBits.One; //停止位
                    serialPort.Parity = Parity.None;
                    serialPort.RtsEnable = false;
                    serialPort.DtrEnable = false;
                    serialPort.Open();   //打开串口
                    //cbSerial.Text = "关闭串口";
                    CbSerial.Enabled = false;
                    UartOpenState = true;
                }
            }
            catch (Exception ex)
            { 
                Console.WriteLine(ex.ToString());
            }
            return UartOpenState;
        }

        /// <summary>
        /// 关闭串口
        /// </summary>
        public void CloseSerial()
        {
            try
            {
                serialPort.Close();  //关闭串口
                //cbSerial.Text = "打开串口";
                CbSerial.Enabled = true;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
        }

        /// <summary>
        /// 生成电池图标
        /// </summary>
        public Image CreateBatteryIcon(int batteryPercentage)
        {
            int width = 18;  // 宽度
            int height = 25; // 高度
            int borderThickness = 2; // 边框厚度
            int batteryTipHeight = 4; // 电池顶端的小突起高度

            Bitmap bitmap = new Bitmap(width, height + batteryTipHeight);
            using (Graphics g = Graphics.FromImage(bitmap))
            {
                g.Clear(Color.White); // 背景色

                // 绘制电池外框
                g.DrawRectangle(new Pen(Color.Black, borderThickness),
                                new Rectangle(0, batteryTipHeight, width, height));

                // 绘制电池顶端的小突起
                g.FillRectangle(Brushes.Black, new Rectangle(width / 3, 0, width / 3, batteryTipHeight));

                // 根据电池电量绘制电池内部
                int fillHeight = (int)((height - 2 * borderThickness) * (batteryPercentage / 100.0));
                g.FillRectangle(Brushes.Green, new Rectangle(borderThickness, height - borderThickness - fillHeight + batteryTipHeight, width - 2 * borderThickness, fillHeight));
            }

            bitmap.RotateFlip(RotateFlipType.Rotate90FlipNone);

            return bitmap;
        }

        /// <summary>
        /// 串口号下拉列表点击事件
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void CbSerial_Click(object sender, EventArgs e)
        {
            //检查是否含有串口
            string[] str = SerialPort.GetPortNames();
            if (str == null)
            {
                MessageBox.Show("本机没有串口！", "Error");
                return;
            }

            string name = null;
            if (CbSerial.SelectedItem != null)
                name = CbSerial.SelectedItem.ToString();

            CbSerial.Items.Clear();

            //更新串口列表
            CbSerial.Items.AddRange(SerialPort.GetPortNames());

            CbSerial.SelectedIndex = CbSerial.FindStringExact(name);
        }

        /// <summary>
        /// 发送指令
        /// </summary>
        /// <param name="cmd"></param>
        public void SendCmd(byte[] cmd) 
        {
            try
            {
                if (serialPort.IsOpen)
                {
                    serialPort.Write(cmd, 0, cmd.Length);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("tx " + ex);
                serialPort.DtrEnable = false;  //解决 “连到系统上的设备没有发挥作用”的问题
                serialPort.Close();
            }
        }

        /// <summary>
        /// 设置信号类型
        /// </summary>
        /// <param name="signalType"></param>
        public void SetCbSignalType(string signalType)
        {
            int index = CbSignalType.Items.IndexOf(signalType);
            if (index >= 0)
            {
                CbSignalType.SelectedIndex = index;
            }
        }

        /// <summary>
        /// 更新信号类型
        /// </summary>
        public void UpdateSignalType()
        {
            if (CbSignalType.SelectedItem != null)
            {
                AdsData.SignalType = CbSignalType.SelectedItem.ToString();
            }
        }

        /// <summary>
        /// 发送采样参数配置指令
        /// </summary>
        public void SendSampleParCmd()
        {
            byte[] cmd = ADS1298_Cmd.SetSampleParCmd(CbRate.Text, CbRange.Text);
            SendCmd(cmd);
        }
    }
}
