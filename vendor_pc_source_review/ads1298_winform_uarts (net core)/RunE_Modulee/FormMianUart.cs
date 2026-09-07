using OxyPlot.Series;
using OxyPlot;
using OxyPlot.WindowsForms;
using RunE_Moudule;
using System.Text;
using Windows.Devices.Bluetooth;
using System.Windows.Forms;
using System.Collections.Generic;
using System;
using System.Linq;
using System.Drawing;
using System.Threading.Tasks;
using System.IO;
using OxyPlot.Axes;
using OxyPlot.Legends;
using System.Diagnostics;
using System.Security.Cryptography;
using RunE_Module;
using Newtonsoft.Json.Linq;
using System.IO.Ports;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace RunE_Modulee
{
    public partial class FormMianUart : Form
    {
        //绘图控件列表
        private Dictionary<string, PlotView> plotViews = new Dictionary<string, PlotView>();

        /// <summary>
        /// 设备操作对象
        /// </summary>
        private Uart uart = new Uart();

        /// 采集状态
        /// </summary>
        private bool SampleState = false;

        public FormMianUart()
        {
            InitializeComponent();
            CheckForIllegalCrossThreadCalls = false;
        }

        private void FormMian_Load(object sender, EventArgs e)
        {
            flowLayoutPanel1.Controls.Add(uart);

            Task t = new Task(async () =>
            {
                while (true)
                {
                    await Task.Delay(50);
                    try
                    {
                        this.Invoke(new Action(() =>
                        {
                            UpdateShow(uart);
                        }));
                    }
                    catch { }
                }
            });
            t.Start();

            RtJson rtJson = new RtJson();
            JObject jsonFile = rtJson.Readjson("config.json");
            if (jsonFile != null)
            {
                if (jsonFile.TryGetValue("SerialPorts", out JToken o_serialPorts))
                {
                    List<string> list = new List<string>(o_serialPorts.ToString().Split(','));
                    if (list.Count > 0)
                    {
                        uart.CbSerial.Text = list[0];
                    }
                }

                if (jsonFile.TryGetValue("SignalTypes", out JToken? o_SignalTypes))
                {
                    List<string> list = new List<string>(o_SignalTypes.ToString().Split(','));
                    if (list.Count > 0)
                    {
                        uart.SetCbSignalType(list[0]);
                    }
                }
            }
        }

        private void FormMian_Resize(object sender, EventArgs e)
        {
            AdjustPlotViewSizes();
        }

        private async void FormMian_FormClosingAsync(object sender, FormClosingEventArgs e)
        {
            if (SampleState)
            {
                var result = MessageBox.Show("采集数据中，是否确定退出？", "确认退出", MessageBoxButtons.YesNo, MessageBoxIcon.Question);
                if (result == DialogResult.Yes)
                {
                    uart.SendCmd(ADS1298_Cmd.StopCollectCmd());
                }
                else
                {
                    e.Cancel = true; // 取消关闭操作
                }
            }
            else
            {
                uart.SendCmd(ADS1298_Cmd.StopCollectCmd());
            }
        }

        /// <summary>
        /// 开始采集
        /// </summary>  
        private void BtnStart_Click(object sender, EventArgs e)
        {
            if (btn_start.Text == "开始采集")
            {
                if (!uart.UartOpenState)
                {
                    MessageBox.Show("无设备连接", "Tips", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }

                uart.AdsData.Clear();
                uart.UpdateSignalType();

                foreach (var plotView in plotViews.Values)
                {
                    foreach (var s in plotView.Model.Series)
                    {
                        (s as LineSeries)?.Points.Clear();
                    }

                    var xAxis = plotView.Model.Axes.FirstOrDefault(axis => axis.Position == AxisPosition.Bottom);
                    if (xAxis != null)
                    {
                        xAxis.Minimum = double.NaN;
                        xAxis.Maximum = double.NaN;
                    }

                    var yAxis = plotView.Model.Axes.FirstOrDefault(axis => axis.Position == AxisPosition.Left);
                    if (yAxis != null)
                    {
                        // Per-channel plots auto-fit their own vertical scale.
                        yAxis.Minimum = double.NaN;
                        yAxis.Maximum = double.NaN;
                        yAxis.Reset();
                    }

                    // 刷新图表
                    plotView.Model.InvalidatePlot(true);
                }

                uart.SendSampleParCmd();
                Thread.Sleep(50);
                uart.SendCmd(ADS1298_Cmd.StartCollectCmd());
                uart.SampleState = true;
                btn_start.Text = "停止采集";
                SampleState = true;
            }
            else
            {
                uart.SampleState = false;
                uart.SendCmd(ADS1298_Cmd.StopCollectCmd());
               
                btn_start.Text = "开始采集";
                SampleState = false;
            }
        }

        /// <summary>
        /// 调整绘图控件的尺寸
        /// </summary>
        private void AdjustPlotViewSizes()
        {

            int plotViewCount = flowLayoutPanel_draw.Controls.Count;
            if (plotViewCount == 0) return;

            //获取排序后的list
            var sortedPlotViews = flowLayoutPanel_draw.Controls.OfType<PlotView>()
                                            .OrderBy(plotView => plotView.Name)
                                            .ToList();


            flowLayoutPanel_draw.Controls.Clear();
            foreach (var plotView in sortedPlotViews)
            {
                flowLayoutPanel_draw.Controls.Add(plotView);
            }

            int fixedWidth = flowLayoutPanel_draw.ClientSize.Width - 10;
            int height = flowLayoutPanel_draw.ClientSize.Height / plotViewCount - 10;
            foreach (Control control in flowLayoutPanel_draw.Controls)
            {
                if (control is PlotView plotView)
                {
                    plotView.Size = new Size(fixedWidth, height);
                }
            }

            flowLayoutPanel_draw.Refresh();
        }

        /// <summary>
        /// 导出数据
        /// </summary>  
        private void BtnExportData_Click(object sender, EventArgs e)
        {
            int maxDataPoints = 0;
            var csvLines = new List<string>();
            var header = new StringBuilder();

            if (SampleState)
            {
                MessageBox.Show("数据采集中无法进行导出操作！！！。", "导出", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            // 构建CSV头部
            int ch_num = uart.AdsData.EmgRaws.Count;
            string ch_str = "";
            for (int ch = 0; ch < ch_num - 1; ch++)
            {
                ch_str += $"CH{ch + 1},";
            }
            ch_str += $"CH{ch_num}";
            csvLines.Add(ch_str);

            try
            {
                maxDataPoints = uart.AdsData.EmgRaws.Max(v => v.Count);
            }
            catch (Exception)
            {
                MessageBox.Show("无采集数据", "Tips");
                return;
            }

            for (int i = 0; i < maxDataPoints; i++)
            {
                var line = new StringBuilder();
                for (int ch = 0; ch < ch_num; ch++)
                {
                    if (i < uart.AdsData.EmgRaws[ch].Count)
                    {
                        line.Append(uart.AdsData.EmgRaws[ch][i]).Append(',');
                    }
                    else
                    {
                        line.Append(',');
                    }
                }
                csvLines.Add(line.ToString().TrimEnd(','));
            }

            SaveFileDialog saveFileDialog = new SaveFileDialog
            {
                Filter = "CSV files (*.csv)|*.csv",
                Title = "保存为 CSV 文件",
                FileName = $"ads129x_{DateTime.Now:yyyyMMdd_HHmmss}.csv"
            };

            if (saveFileDialog.ShowDialog() == DialogResult.OK)
            {
                File.WriteAllLines(saveFileDialog.FileName, csvLines);
                MessageBox.Show("数据导出成功。", "导出", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
        }

        /// <summary>
        /// 添加绘制控件
        /// </summary>
        private void AddPlotView(string number)
        {
            if (this.InvokeRequired)
            {
                this.Invoke(new Action<string>(AddPlotView), number);
            }
            else
            {
                // One plot per channel: eight stacked PlotViews, each auto-fit.
                for (int ch = 0; ch < 8; ch++)
                {
                    string plotViewName = $"{number}_CH{ch + 1}";
                    var existingPlotView = flowLayoutPanel_draw.Controls
                    .OfType<PlotView>()
                    .FirstOrDefault(pv => pv.Name == plotViewName);

                    if (existingPlotView != null)
                    {
                        flowLayoutPanel_draw.Controls.Remove(existingPlotView);
                        plotViews.Remove(plotViewName);
                    }

                    PlotView plotView = new PlotView
                    {
                        Name = plotViewName,
                        BackColor = Color.FromArgb(0xFF, 0xFF, 0xFF),
                    };

                    var plotModel = new PlotModel
                    {
                        Title = $"CH{ch + 1}",
                    };

                    // Y axis auto-fits the signal so every channel stays visible.
                    plotModel.Axes.Add(new LinearAxis
                    {
                        Title = "uV",
                        TitleFontSize = 12,
                        FontSize = 10,
                        MajorGridlineStyle = LineStyle.Solid,
                        MinorGridlineStyle = LineStyle.Dot,
                        TickStyle = OxyPlot.Axes.TickStyle.Inside,
                        Minimum = double.NaN,
                        Maximum = double.NaN,
                        Position = AxisPosition.Left,
                    });

                    // X axis rolls a 5-second window (rate * 5 points).
                    plotModel.Axes.Add(new LinearAxis
                    {
                        MajorGridlineStyle = LineStyle.Solid,
                        MinorGridlineStyle = LineStyle.Dot,
                        TickStyle = OxyPlot.Axes.TickStyle.Inside,
                        IntervalLength = 80,
                        Position = AxisPosition.Bottom,
                    });

                    var series1 = new LineSeries
                    {
                        Title = $"CH{ch + 1}",
                        EdgeRenderingMode = EdgeRenderingMode.PreferSpeed,
                    };
                    plotModel.Series.Add(series1);

                    plotView.Model = plotModel;
                    flowLayoutPanel_draw.Controls.Add(plotView);
                    plotViews[plotViewName] = plotView;
                }
                AdjustPlotViewSizes();
            }
        }

        /// <summary>
        /// 打开串口
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void BtnSerial_Click(object sender, EventArgs e)
        {
            if (BtnSerial.Text == "打开串口")
            {
                List<string> serialPorts = new List<string>();
                List<string> signalTypes = new List<string>();

                flowLayoutPanel_draw.Controls.Clear();
                plotViews.Clear();

                if (uart.OpenSerial())
                {
                    AddPlotView(uart.Number.Text);
                }
                serialPorts.Add(uart.CbSerial.Text);
                signalTypes.Add(uart.CbSignalType.Text);

                JObject serialPortsJson = new JObject();
                serialPortsJson["SerialPorts"] = string.Join(",", serialPorts.ToArray());
                serialPortsJson["SignalTypes"] = string.Join(",", signalTypes.ToArray());
                RtJson rtJson = new RtJson();
                rtJson.Writejson("config.json", serialPortsJson);

                BtnSerial.Text = "关闭串口";
            }
            else
            {
                uart.CloseSerial();
                BtnSerial.Text = "打开串口";
            }
        }

        /// <summary>
        /// 更新绘图
        /// </summary>
        /// <param name="uart"></param>
        private void UpdateShow(Uart uart)
        {
            string keyPrefix = uart.Number.Text;
            int rate = uart.AdsData.Rate;

            if (SampleState == false)
                return;

            for (int ch = 0; ch < 8; ch++)
            {
                if (!plotViews.TryGetValue($"{keyPrefix}_CH{ch + 1}", out PlotView plotView))
                    continue;

                LineSeries? line = plotView.Model.Series[0] as LineSeries;
                if (line == null)
                    continue;

                int x0 = 0;
                if (line.Points.Count > 0)
                {
                    DataPoint dataPoint = line.Points.Last();
                    x0 = (int)(dataPoint.X);
                }
                int x1 = uart.AdsData.EmgRaws[ch].Count;

                if ((x1 - x0) < 100)
                    continue;

                for (int i = x0; i < x1; i++)
                {
                    line.Points.Add(new DataPoint(i, uart.AdsData.EmgRaws[ch][i]));
                }

                int dataShowLen = rate * 5;
                if ((x1 - dataShowLen) > 0)
                {
                    plotView.Model.Axes[1].Minimum = x1 - dataShowLen;
                    plotView.Model.Axes[1].Maximum = x1;
                }

                int dataShowMaxLen = rate * 120;
                if (line.Points.Count > dataShowMaxLen)
                {
                    line.Points.RemoveRange(0, line.Points.Count - dataShowMaxLen);
                }
                plotView.Model.InvalidatePlot(true);
            }

        }

        /// <summary>
        /// 滤波器打开关闭事件
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void CbFilter_CheckedChanged(object sender, EventArgs e)
        {
            uart.AdsData.FiltrSwitch = CbFilter.Checked;
        }
    }
}
