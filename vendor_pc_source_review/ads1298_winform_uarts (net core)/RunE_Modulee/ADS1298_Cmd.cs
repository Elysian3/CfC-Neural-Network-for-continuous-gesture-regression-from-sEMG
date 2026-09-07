using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RunE_Moudule
{
    public class ADS1298_Cmd
    {
        public const byte ADDRESS_BOARD = 0x01;          //获取PCB硬件版本
        public const byte ADDRESS_SOFTWARE = 0x02;       //获取软件版本
        public const byte ADDRESS_CONN_STATUS_UPDATE = 0x09; //连接状态更新
        public const byte ADDRESS_SMAPLE_PAR = 0x10;     //采样参数配置
        public const byte ADDRESS_START = 0x11;      //开始/停止采集指令(采集原始肌电数据)

        /// <summary>
        /// 指令帧打包
        /// </summary>
        /// <param name="address"></param>
        /// <param name="isWrite"></param>
        /// <param name="data"></param>
        /// <returns></returns>
        private static byte[] CmdDataPack(byte address, bool isWrite, List<byte> data)
        {
            List<byte> cmd = new List<byte>();

            cmd.Add(0xAA);
            cmd.Add((byte)(data.Count + 3));

            if (isWrite)
            {
                cmd.Add(0x80); //写指令
            }
            else
            {
                cmd.Add(0x81); //读指令
            }

            //地址
            cmd.Add(address); //3

            //命令内容
            cmd.AddRange(data);
            
            //校验码（待定）
            cmd.Add(0x00);

            //帧尾
            cmd.Add(0xBB);

            //计算校验码
            byte xor = 0x00;
            for (int i = 1; i < cmd.Count - 2; i++)
            {
                xor ^= cmd[i];
            }
            cmd[cmd.Count - 2] = xor;

            return cmd.ToArray();
        }

        /// <summary>
        /// 获取硬件版本信息
        /// </summary>
        /// <returns></returns>
        public static byte[] GetHardwareVersionCmd()
        {
            return CmdDataPack(ADDRESS_BOARD, false, new List<byte>());
        }

        /// <summary>
        /// 获取软件版本信息
        /// </summary>
        /// <returns></returns>
        public static byte[] GetSoftwareVersioCmd()
        {
            return CmdDataPack(ADDRESS_SOFTWARE, false, new List<byte>());
        }

        /// <summary>
        /// 停止采集
        /// </summary>
        /// <returns></returns>
        public static byte[] StopCollectCmd()
        {
            List<byte> listData = new List<byte>();
            listData.Add(0x00);
            return CmdDataPack(ADDRESS_START, true, listData);
        }

        /// <summary>
        /// 开始采集
        /// </summary>
        /// <returns></returns>
        public static byte[] StartCollectCmd()
        {
            List<byte> listData = new List<byte>();
            listData.Add(0x01);
            return CmdDataPack(ADDRESS_START, true, listData);
        }

        /// <summary>
        /// 连接状态更新
        /// 0: 断开连接
        /// 1: 已连接
        /// 2: 保持连接
        /// </summary>
        /// <returns></returns>
        public static byte[] ConnStatusUpdate(byte status)
        {
            List<byte> listData = new List<byte>();
            listData.Add(status);
            return CmdDataPack(ADDRESS_CONN_STATUS_UPDATE, true, listData);
        }

        /// <summary>
        /// 设置采样参数设置
        /// </summary>
        /// <returns></returns>
        public static byte[] SetSampleParCmd(string rate, string range)
        {
            string[] rates = new string[] { "500sps", "1000sps", "2000sps", "4000sps" };
            string[] ranges = new string[] { "±2.4V", "±1.2V", "±800mV", "±600mV", "±400mV", "±300mV", "±200mV" };

            int rate_index = Array.IndexOf(rates, rate);
            int range_index = Array.IndexOf(ranges, range);

            rate_index = rate_index < 0 ? 0 : rate_index;
            range_index = range_index < 0 ? 0 : range_index;

            List<byte> listData = new List<byte>();
            listData.Add((byte)rate_index);
            listData.Add((byte)range_index);
            listData.Add((byte)0x00);
            return CmdDataPack(ADDRESS_SMAPLE_PAR, true, listData);
        }
    }
}
