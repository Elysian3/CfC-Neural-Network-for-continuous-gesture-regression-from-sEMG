using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RunE_Module
{
    public class EmgFilter
    {
        /// <summary>
        /// 滤波器集合，key为采样率
        /// </summary>
        private Dictionary<int,List<IIRFilter>> DictFilter = new Dictionary<int,List<IIRFilter>>();

        public EmgFilter()
        {
            AddFilter500();
            AddFilter1000();
            AddFilter2000();
        }

        /// <summary>
        /// 添加采样率500Hz的滤波器系数
        /// </summary>
        public void AddFilter500()
        {
            List<IIRFilter> filters = new List<IIRFilter>();

            // 高通滤波器（6阶，20Hz）
            double[] num_h = new double[] { 0.614371537726035, - 3.68622922635621  , 9.21557306589052 ,- 12.2874307545207  , 9.21557306589052 ,- 3.68622922635621 ,  0.614371537726035 };
            double[] den_h = new double[] { 1 ,- 5.02943835142161  , 10.6070421837797 ,- 11.9993158162167  , 7.67547454820020 ,- 2.63105512847395 ,  0.377452386374089 };
            filters.Add(new IIRFilter(num_h, den_h));


            // 50Hz工频陷滤器
            double[] num_noth = new double[] { 0.984260526926093, - 1.59256698635130  , 0.984260526926093 };
            double[] den_noth = new double[] { 1 ,- 1.59256698635130 ,  0.968521053852186 };
            filters.Add(new IIRFilter(num_noth, den_noth));


            DictFilter.Add(500, filters);
        }

        /// <summary>
        /// 添加采样率1000Hz的滤波器系数
        /// </summary>
        public void AddFilter1000()
        {
            List<IIRFilter> filters = new List<IIRFilter>();

            // 高通滤波器（6阶，20Hz）
            double[] num_h = new double[] { 0.784297852893036, - 4.70578711735822,   11.7644677933955, - 15.6859570578607  , 11.7644677933955 ,- 4.70578711735822   ,0.784297852893036 };
            double[] den_h = new double[] { 1, - 5.51453512116617  , 12.6891130565151 ,- 15.5936352107041  , 10.7932966704854 ,- 3.98935940423088 ,  0.615123122052628 };
            filters.Add(new IIRFilter(num_h, den_h));
            DictFilter.Add(1000, filters);
        }

        /// <summary>
        /// 添加采样率2000Hz的滤波器系数
        /// </summary>
        public void AddFilter2000()
        {
            List<IIRFilter> filters = new List<IIRFilter>();

            // 高通滤波器（6阶，20Hz）
            double[] num_h = new double[] { 0.885673290152356 ,-5.31403974091414 ,  13.2850993522853, -17.7134658030471  , 13.2850993522853 ,-5.31403974091414   ,0.885673290152356 };
            double[] den_h = new double[] { 1 ,-5.75724418624657 , 13.8155108060580, -17.6873761798940  , 12.7416173292292 ,- 4.89692489143373 , 0.784417176889300 };
            filters.Add(new IIRFilter(num_h, den_h));
            DictFilter.Add(2000, filters);
        }

        /// <summary>
        /// 数据滤波
        /// </summary>
        /// <param name="rate"></param>
        /// <param name="input"></param>
        /// <returns></returns>
        public float ProcessSample(int rate,float input)
        {
            double output = input;

            if (DictFilter.TryGetValue(rate, out List<IIRFilter>? filters))
            {
                foreach (var filter in filters)
                {
                    output = filter.ProcessSample(output);
                }
            }

            return (float)output;
        }
    }
}
