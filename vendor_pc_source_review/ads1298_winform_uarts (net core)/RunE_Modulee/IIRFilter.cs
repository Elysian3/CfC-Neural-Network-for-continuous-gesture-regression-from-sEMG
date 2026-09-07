using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RunE_Module
{
    public class IIRFilter
    {
        private double[] a; // 分母系数
        private double[] b; // 分子系数
        private double[] x; // 输入历史
        private double[] y; // 输出历史

        public IIRFilter(double[] numerator, double[] denominator)
        {
            if (numerator == null || denominator == null || denominator.Length == 0)
            {
                throw new ArgumentException("Numerator and denominator coefficients must not be null or empty.");
            }

            this.b = numerator;
            this.a = denominator;

            // 初始化输入和输出历史缓冲区
            this.x = new double[b.Length];
            this.y = new double[a.Length - 1]; // a[0] 是归一化系数，不需要存储
        }

        public double ProcessSample(double input)
        {
            // 更新输入历史
            for (int i = x.Length - 1; i > 0; i--)
            {
                x[i] = x[i - 1];
            }
            x[0] = input;

            // 计算输出
            double output = 0.0;

            // 计算分子部分
            for (int i = 0; i < b.Length; i++)
            {
                output += b[i] * x[i];
            }

            // 计算分母部分
            for (int i = 0; i < y.Length; i++)
            {
                output -= a[i + 1] * y[i];
            }

            // 归一化
            output /= a[0];

            // 更新输出历史
            for (int i = y.Length - 1; i > 0; i--)
            {
                y[i] = y[i - 1];
            }
            y[0] = output;

            return output;
        }
    }

    //class Program
    //{
    //    static void Main(string[] args)
    //    {
    //        // 示例：二阶低通滤波器
    //        double[] numerator = { 1.0, 2.0, 1.0 }; // 分子系数
    //        double[] denominator = { 1.0, -1.6, 0.7 }; // 分母系数

    //        IIRFilter filter = new IIRFilter(numerator, denominator);

    //        // 示例输入信号
    //        double[] inputSignal = { 1.0, 0.5, 0.2, 0.1, 0.0 };
    //        double[] outputSignal = new double[inputSignal.Length];

    //        // 处理每个采样点
    //        for (int i = 0; i < inputSignal.Length; i++)
    //        {
    //            outputSignal[i] = filter.ProcessSample(inputSignal[i]);
    //            //Console.WriteLine($"Input: {inputSignal[i]}, Output: {outputSignal[i]}");
    //        }
    //    }
    //}
}
