using System.Drawing;
using System.Windows.Forms;

namespace RunE_Modulee
{
    partial class FormMianUart
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(FormMianUart));
            flowLayoutPanel_draw = new FlowLayoutPanel();
            backgroundWorker1 = new System.ComponentModel.BackgroundWorker();
            contextMenuStrip1 = new ContextMenuStrip(components);
            断开连接ToolStripMenuItem = new ToolStripMenuItem();
            panel2 = new Panel();
            BtnSerial = new Button();
            btn_start = new Button();
            btn_export = new Button();
            label1 = new Label();
            flowLayoutPanel1 = new FlowLayoutPanel();
            panel1 = new Panel();
            CbFilter = new CheckBox();
            colorDialog1 = new ColorDialog();
            contextMenuStrip1.SuspendLayout();
            panel2.SuspendLayout();
            panel1.SuspendLayout();
            SuspendLayout();
            // 
            // flowLayoutPanel_draw
            // 
            flowLayoutPanel_draw.AutoScroll = true;
            flowLayoutPanel_draw.BackColor = Color.White;
            flowLayoutPanel_draw.Dock = DockStyle.Fill;
            flowLayoutPanel_draw.FlowDirection = FlowDirection.TopDown;
            flowLayoutPanel_draw.Location = new Point(0, 0);
            flowLayoutPanel_draw.Margin = new Padding(4);
            flowLayoutPanel_draw.Name = "flowLayoutPanel_draw";
            flowLayoutPanel_draw.Size = new Size(859, 618);
            flowLayoutPanel_draw.TabIndex = 22;
            flowLayoutPanel_draw.Tag = "";
            flowLayoutPanel_draw.WrapContents = false;
            // 
            // contextMenuStrip1
            // 
            contextMenuStrip1.Items.AddRange(new ToolStripItem[] { 断开连接ToolStripMenuItem });
            contextMenuStrip1.Name = "contextMenuStrip1";
            contextMenuStrip1.Size = new Size(125, 26);
            // 
            // 断开连接ToolStripMenuItem
            // 
            断开连接ToolStripMenuItem.Name = "断开连接ToolStripMenuItem";
            断开连接ToolStripMenuItem.Size = new Size(124, 22);
            断开连接ToolStripMenuItem.Text = "断开连接";
            // 
            // panel2
            // 
            panel2.Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right;
            panel2.Controls.Add(flowLayoutPanel_draw);
            panel2.Location = new Point(338, 0);
            panel2.Margin = new Padding(4);
            panel2.Name = "panel2";
            panel2.Size = new Size(859, 618);
            panel2.TabIndex = 29;
            // 
            // BtnSerial
            // 
            BtnSerial.BackColor = Color.FromArgb(5, 71, 168);
            BtnSerial.Font = new Font("宋体", 12F, FontStyle.Regular, GraphicsUnit.Point, 134);
            BtnSerial.ForeColor = Color.White;
            BtnSerial.ImageAlign = ContentAlignment.MiddleLeft;
            BtnSerial.Location = new Point(15, 245);
            BtnSerial.Margin = new Padding(4);
            BtnSerial.Name = "BtnSerial";
            BtnSerial.Size = new Size(224, 46);
            BtnSerial.TabIndex = 13;
            BtnSerial.Tag = "扫描";
            BtnSerial.Text = "打开串口";
            BtnSerial.UseVisualStyleBackColor = false;
            BtnSerial.Click += BtnSerial_Click;
            // 
            // btn_start
            // 
            btn_start.BackColor = Color.FromArgb(5, 71, 168);
            btn_start.Font = new Font("宋体", 12F, FontStyle.Regular, GraphicsUnit.Point, 134);
            btn_start.ForeColor = SystemColors.Control;
            btn_start.Location = new Point(15, 306);
            btn_start.Margin = new Padding(4);
            btn_start.Name = "btn_start";
            btn_start.Size = new Size(224, 46);
            btn_start.TabIndex = 21;
            btn_start.Text = "开始采集";
            btn_start.UseVisualStyleBackColor = false;
            btn_start.Click += BtnStart_Click;
            // 
            // btn_export
            // 
            btn_export.BackColor = Color.FromArgb(5, 71, 168);
            btn_export.Font = new Font("宋体", 12F, FontStyle.Regular, GraphicsUnit.Point, 134);
            btn_export.ForeColor = Color.White;
            btn_export.ImageAlign = ContentAlignment.MiddleLeft;
            btn_export.Location = new Point(12, 367);
            btn_export.Margin = new Padding(4);
            btn_export.Name = "btn_export";
            btn_export.Size = new Size(227, 46);
            btn_export.TabIndex = 23;
            btn_export.Text = "导出数据";
            btn_export.UseVisualStyleBackColor = false;
            btn_export.Click += BtnExportData_Click;
            // 
            // label1
            // 
            label1.AutoSize = true;
            label1.Location = new Point(18, 18);
            label1.Margin = new Padding(4, 0, 4, 0);
            label1.Name = "label1";
            label1.Size = new Size(56, 17);
            label1.TabIndex = 27;
            label1.Text = "连接列表";
            // 
            // flowLayoutPanel1
            // 
            flowLayoutPanel1.Location = new Point(12, 38);
            flowLayoutPanel1.Name = "flowLayoutPanel1";
            flowLayoutPanel1.Size = new Size(302, 105);
            flowLayoutPanel1.TabIndex = 31;
            // 
            // panel1
            // 
            panel1.Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left;
            panel1.BackColor = Color.White;
            panel1.Controls.Add(CbFilter);
            panel1.Controls.Add(flowLayoutPanel1);
            panel1.Controls.Add(label1);
            panel1.Controls.Add(btn_export);
            panel1.Controls.Add(btn_start);
            panel1.Controls.Add(BtnSerial);
            panel1.Location = new Point(3, 0);
            panel1.Margin = new Padding(4);
            panel1.Name = "panel1";
            panel1.Size = new Size(327, 618);
            panel1.TabIndex = 28;
            // 
            // CbFilter
            // 
            CbFilter.AutoSize = true;
            CbFilter.Checked = true;
            CbFilter.CheckState = CheckState.Checked;
            CbFilter.Font = new Font("Microsoft YaHei UI", 12F, FontStyle.Regular, GraphicsUnit.Point, 134);
            CbFilter.Location = new Point(74, 184);
            CbFilter.Name = "CbFilter";
            CbFilter.Size = new Size(109, 25);
            CbFilter.TabIndex = 32;
            CbFilter.Text = "开启滤波器";
            CbFilter.UseVisualStyleBackColor = true;
            CbFilter.CheckedChanged += CbFilter_CheckedChanged;
            // 
            // FormMianUart
            // 
            AutoScaleDimensions = new SizeF(7F, 17F);
            AutoScaleMode = AutoScaleMode.Font;
            BackColor = SystemColors.Control;
            ClientSize = new Size(1195, 614);
            Controls.Add(panel2);
            Controls.Add(panel1);
            Icon = (Icon)resources.GetObject("$this.Icon");
            Margin = new Padding(4);
            Name = "FormMianUart";
            Text = "ADS1298_Module（串口版本）";
            FormClosing += FormMian_FormClosingAsync;
            Load += FormMian_Load;
            Resize += FormMian_Resize;
            contextMenuStrip1.ResumeLayout(false);
            panel2.ResumeLayout(false);
            panel1.ResumeLayout(false);
            panel1.PerformLayout();
            ResumeLayout(false);
        }

        #endregion
        private System.Windows.Forms.FlowLayoutPanel flowLayoutPanel_draw;
        private System.ComponentModel.BackgroundWorker backgroundWorker1;
        private System.Windows.Forms.Panel panel2;
        private ContextMenuStrip contextMenuStrip1;
        private ToolStripMenuItem 断开连接ToolStripMenuItem;
        private Button btn_start;
        private Button btn_export;
        private Label label1;
        private Button BtnSerial;
        private FlowLayoutPanel flowLayoutPanel1;
        private Panel panel1;
        private ColorDialog colorDialog1;
        private CheckBox CbFilter;
    }
}
