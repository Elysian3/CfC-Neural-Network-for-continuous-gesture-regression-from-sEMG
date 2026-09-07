using RunE_Module;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing.Drawing2D;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Windows.Media.Core;
using static System.Net.WebRequestMethods;

namespace RunE_Moudule
{
    public class ADS1298_Data
    {
        /// <summary>
        /// EMG通道数量
        /// </summary>
        private const int EMG_CH_NUM = 8;

        /// <summary>
        /// 数据缓存
        /// </summary>
        private List<byte> rxBuffer = new List<byte>();

        /// <summary>
        /// 肌电信号
        /// </summary>
        public List<List<float>> EmgRaws= new List<List<float>>();

        /// <summary>
        /// 电池电量
        /// </summary>
        public float BatteryLevel = 100;

        /// <summary>
        /// 采样率
        /// </summary>
        public int Rate = 500;

        /// <summary>
        /// 信号类型:EMG、EEG、ECG
        /// </summary>
        public string SignalType = "EMG";

        /// <summary>
        /// 肌电滤波器
        /// </summary>
        private List<EmgFilter> emgFilters = new List<EmgFilter>();

        /// <summary>
        /// 心电滤波器
        /// </summary>
        private List<EcgFilter> ecgFilters = new List<EcgFilter>();

        /// <summary>
        /// 脑电滤波器
        /// </summary>
        private List<EegFilter> eegFilters = new List<EegFilter>();

        /// <summary>
        /// 肌电滤波器开关
        /// </summary>
        public bool FiltrSwitch = true;

        public ADS1298_Data() 
        {
            for (int ch = 0; ch < EMG_CH_NUM; ch++)
            {
                EmgRaws.Add(new List<float>());
                emgFilters.Add(new EmgFilter());
                ecgFilters.Add(new EcgFilter());
                eegFilters.Add(new EegFilter());
            }
        }


        public void Clear()
        {
            rxBuffer.Clear();

            foreach (List<float> val in EmgRaws)
            {
                val.Clear();
            }
        }

        /// <summary>
        /// 缓存数据
        /// </summary>
        /// <param name="data"></param>
        public void AddData(byte[] data)
        {
            rxBuffer.AddRange(data);
            if (rxBuffer.Count > 5)
            {
                DataUnpack();
            }
        }

        /// <summary>
        /// 数据解析
        /// </summary>
        private void DataUnpack()
        {
            while (rxBuffer.Count > 5)
            {
                var checkByte = ((rxBuffer[1]) ^ (rxBuffer[2]));
                //帧头校验
                if ((rxBuffer[0] == 0xa5) && (checkByte == rxBuffer[3]))
                {
                    //帧长度
                   int frameLen = (rxBuffer[1] & 0xFF) + 3;
                    //判断剩于数据长度是否大于帧长度
                    if (frameLen <= (rxBuffer.Count))
                    {
                        //校验帧尾
                        if ((rxBuffer[frameLen - 1] & 0xFF) == 0x5A)
                        {
                            //var dataFrame = _rxUartData.sublist(0, frameLen);
                             byte[] dataFrame = rxBuffer.GetRange(0, frameLen).ToArray();
                            FrameUnpack(dataFrame);
                            //移除已解包数据
                            rxBuffer.RemoveRange(0, frameLen);
                        }
                        else
                        {
                            //帧尾不对，指针后移
                            rxBuffer.RemoveAt(0);
                        }
                    }
                    else
                    {
                        //帧不完整，退出解码
                        // LogD("剩于数据长度不足");
                        break;
                    }
                }
                else
                {
                    // LogE("帧头出错 ${_rxUartData[0]}");
                    //不是帧头，指针后移
                    rxBuffer.RemoveAt(0);
                }
            }
        }

        /// <summary>
        /// 数据帧解析
        /// </summary>
        /// <param name="frame"></param>
        private void FrameUnpack(byte[] frame)
        {
            //帧地址/类型
            byte frameType = frame[2];
            switch (frameType & 0xFF)
            {
                case ADS1298_Cmd.ADDRESS_START:
                    FrameEmgRaw(frame);
                    break;
                case ADS1298_Cmd.ADDRESS_SMAPLE_PAR:
                    FrameSamplePar(frame);
                    break;
                default:
                    break;
            }
        }

        /// <summary>
        /// 解析原始数据帧
        /// </summary>
        /// <param name="frame"></param>
        private void FrameEmgRaw(byte[] frame)
        {
            //string timestamp = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss.fff",
            //                                CultureInfo.InvariantCulture);

            int n = (frame[1] - 2) / 4; //数据点数
            float[] emgData = new float[n];

            Buffer.BlockCopy(frame, 4, emgData, 0, n * 4);

            for (int i = 0; i < n / 8; i++)
            {
                int index = i*8;
                for (int ch = 0; ch < 8; ch++)
                {
                    float val = emgData[index+ch];

                    // 数字滤波器
                    if (FiltrSwitch)
                    {
                        switch (SignalType)
                        {
                            // 肌电信号滤波
                            case "EMG":val = emgFilters[ch].ProcessSample(Rate, val);break;

                            // 心电信号滤波
                            case "ECG": val = ecgFilters[ch].ProcessSample(Rate, val); break;

                            // 脑电信号滤波
                            case "EEG": val = eegFilters[ch].ProcessSample(Rate, val); break;

                        }
                    }

                    EmgRaws[ch].Add(val);
                }
            }
        }

        /// <summary>
        /// 采样参数数据帧
        /// </summary>
        /// <param name="frame"></param>
        private void FrameSamplePar(byte[] frame)
        {
            Trace.WriteLine("FrameSamplePar");

            int[] rates = new int[] {500, 1000, 2000,4000};
            byte rate_index = frame[4];
            if (rate_index < rates.Length)
            {
                Rate = rates[rate_index];
                Trace.WriteLine($"Rate : {Rate}");
            }

            int[] pgas = new int[] {1,2,3,4,6,8,12};
            byte pga_index = frame[5];
            if (pga_index < pgas.Length)
            {
                int pga = pgas[pga_index];
                Trace.WriteLine($"pga : {pga}");
            }

        }
    }
}
