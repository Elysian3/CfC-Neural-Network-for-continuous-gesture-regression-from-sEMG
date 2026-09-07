using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Windows.Forms.VisualStyles.VisualStyleElement.Rebar;

namespace RunE_Module
{
    public class EcgFilter
    {
        /// <summary>
        /// 滤波器集合，key为采样率
        /// </summary>
        private Dictionary<int, List<IIRFilter>> DictFilter = new Dictionary<int, List<IIRFilter>>();

        public EcgFilter()
        {
            AddFilter125();
            AddFilter250();
            AddFilter500();
            AddFilter1000();
            AddFilter2000();
        }

        /// <summary>
        /// 添加采样率125Hz的滤波器系数
        /// </summary>
        public void AddFilter125()
        {
            List<IIRFilter> filters = new List<IIRFilter>();

            // 高通滤波器（2阶，0.5Hz）
            double[] num_h = new double[] { 0.982385438526092, -1.96477087705218, 0.982385438526092 };
            double[] den_h = new double[] { 1, -1.96446058020523, 0.965081173899135 };
            filters.Add(new IIRFilter(num_h, den_h));

            // 单点50Hz陷波器
            double[] num_noth = new double[] { 0.911897419411676, 1.47548101886141, 0.911897419411676 };
            double[] den_noth = new double[] { 1, 1.47548101886141, 0.823794838823351 };
            filters.Add(new IIRFilter(num_noth, den_noth));

            DictFilter.Add(125, filters);
        }

        /// <summary>
        /// 添加采样率250Hz的滤波器系数
        /// </summary>
        public void AddFilter250()
        {
            List<IIRFilter> filters = new List<IIRFilter>();

            // 高通滤波器（2阶，0.5Hz）
            double[] num_h = new double[] { 0.991153595101663, -1.98230719020333, 0.991153595101663 };
            double[] den_h = new double[] { 1, -1.98222892979253, 0.982385450614125 };
            filters.Add(new IIRFilter(num_h, den_h));


            // 梳状50Hz陷波器
            double[] num_comb = new double[] { 0.872683978124476, 0, 0, 0, 0, -0.872683978124476 };
            double[] den_comb = new double[] { 1, 0, 0, 0, 0, -0.745367956248952 };
            filters.Add(new IIRFilter(num_comb, den_comb));

            DictFilter.Add(250, filters);
        }

        /// <summary>
        /// 添加采样率500Hz的滤波器系数
        /// </summary>
        public void AddFilter500()
        {
            List<IIRFilter> filters = new List<IIRFilter>();

            // 高通滤波器（4阶，0.5Hz）
            double[] num_h = new double[] { 0.991824212000533, -3.96729684800213, 5.95094527200320, -3.96729684800213, 0.991824212000533 };
            double[] den_h = new double[] { 1, -3.98358125865852, 5.95087842926670, -3.95101243657283, 0.983715267510479 };
            filters.Add(new IIRFilter(num_h, den_h));

            // 低通滤波器（4阶，100Hz）
            double[] num_low = new double[] { 0.0465829066364437, 0.186331626545775, 0.279497439818662, 0.186331626545775, 0.0465829066364437 };
            double[] den_low = new double[] { 1, -0.782095198023338, 0.679978526916300, -0.182675697753033, 0.0301188750431693 };
            filters.Add(new IIRFilter(num_low, den_low));

            // 单点50Hz陷波器
            double[] num_noth_50hz = new double[] { 0.976529811793961, -1.58005842651017, 0.976529811793961 };
            double[] den_noth_50hz = new double[] { 1, -1.58005842651017, 0.953059623587922 };
            filters.Add(new IIRFilter(num_noth_50hz, den_noth_50hz));

            // 单点100Hz陷波器
            double[] num_noth_100hz = new double[] { 0.976529811793961, -0.603528614716206, 0.976529811793961 };
            double[] den_noth_100hz = new double[] { 1, -0.603528614716206, 0.953059623587922 };
            filters.Add(new IIRFilter(num_noth_100hz, den_noth_100hz));

            DictFilter.Add(500, filters);
        }

        /// <summary>
        /// 添加采样率1000Hz的滤波器系数
        /// </summary>
        public void AddFilter1000()
        {
            List<IIRFilter> filters = new List<IIRFilter>();

            // 高通滤波器（4阶，0.5Hz）
            double[] num_h = new double[] { 0.995903722138425, -3.98361488855370, 5.97542233283055, -3.98361488855370, 0.995903722138425 };
            double[] den_h = new double[] { 1, -3.99179062441447, 5.97540555338674, -3.97543915264442, 0.991824223769170 };
            filters.Add(new IIRFilter(num_h, den_h));

            // 低通滤波器（4阶，100Hz）
            double[] num_low = new double[] { 0.00482434335771623, 0.0192973734308649, 0.0289460601462974, 0.0192973734308649, 0.00482434335771623 };
            double[] den_low = new double[] { 1, -2.36951300718204, 2.31398841441588, -1.05466540587857, 0.187379492368185 };
            filters.Add(new IIRFilter(num_low, den_low));

            // 单点50Hz陷波器
            double[] num_noth_50hz = new double[] { 0.988128453800010, -1.87953200984631, 0.988128453800010 };
            double[] den_noth_50hz = new double[] { 1, -1.87953200984631, 0.976256907600020 };
            filters.Add(new IIRFilter(num_noth_50hz, den_noth_50hz));

            // 单点100Hz陷波器
            double[] num_noth_100hz = new double[] { 0.988128453800010, -1.59882542349930, 0.988128453800010 };
            double[] den_noth_100hz = new double[] { 1, -1.59882542349930, 0.976256907600020 };
            filters.Add(new IIRFilter(num_noth_100hz, den_noth_100hz));

            DictFilter.Add(1000, filters);
        }

        /// <summary>
        /// 添加采样率2000Hz的滤波器系数
        /// </summary>
        public void AddFilter2000()
        {
            List<IIRFilter> filters = new List<IIRFilter>();

            // 高通滤波器（4阶，0.5Hz）
            double[] num_h = new double[] { 0.997949760065881, -3.99179904026353, 5.98769856039529, -3.99179904026353, 0.997949760065881 };
            double[] den_h = new double[] { 1, -3.99589531159288, 5.98769435691454, -3.98770276893113, 0.995903723615550 };
            filters.Add(new IIRFilter(num_h, den_h));

            // 低通滤波器（4阶，100Hz）
            double[] num_low = new double[] { 0.000416599204406599, 0.00166639681762640, 0.00249959522643960, 0.00166639681762640, 0.000416599204406599 };
            double[] den_low = new double[] { 1, -3.18063854887472, 3.86119434899421, -2.11215535511097, 0.438265142261980 };
            filters.Add(new IIRFilter(num_low, den_low));

            // 单点50Hz陷波器
            double[] num_noth_50hz = new double[] { 0.994029149261888, -1.96358200187534, 0.994029149261888 };
            double[] den_noth_50hz = new double[] { 1, -1.96358200187534, 0.988058298523775 };
            filters.Add(new IIRFilter(num_noth_50hz, den_noth_50hz));

            // 单点100Hz陷波器
            double[] num_noth_100hz = new double[] { 0.994029149261888, -1.89075579958569, 0.994029149261888 };
            double[] den_noth_100hz = new double[] { 1, -1.89075579958569, 0.988058298523775 };
            filters.Add(new IIRFilter(num_noth_100hz, den_noth_100hz));


            DictFilter.Add(2000, filters);
        }

        /// <summary>
        /// 数据滤波
        /// </summary>
        /// <param name="rate"></param>
        /// <param name="input"></param>
        /// <returns></returns>
        public float ProcessSample(int rate, float input)
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
