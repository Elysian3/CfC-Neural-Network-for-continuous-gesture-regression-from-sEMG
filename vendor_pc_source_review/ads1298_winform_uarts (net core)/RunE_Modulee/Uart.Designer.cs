namespace RunE_Module
{
    partial class Uart
    {
        /// <summary> 
        /// 必需的设计器变量。
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary> 
        /// 清理所有正在使用的资源。
        /// </summary>
        /// <param name="disposing">如果应释放托管资源，为 true；否则为 false。</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region 组件设计器生成的代码

        /// <summary> 
        /// 设计器支持所需的方法 - 不要修改
        /// 使用代码编辑器修改此方法的内容。
        /// </summary>
        private void InitializeComponent()
        {
            components = new System.ComponentModel.Container();
            CbSerial = new ComboBox();
            pictureBox1 = new PictureBox();
            lbConnState = new Label();
            Number = new Label();
            toolTip1 = new ToolTip(components);
            CbSignalType = new ComboBox();
            CbRate = new ComboBox();
            CbRange = new ComboBox();
            ((System.ComponentModel.ISupportInitialize)pictureBox1).BeginInit();
            SuspendLayout();
            // 
            // CbSerial
            // 
            CbSerial.FormattingEnabled = true;
            CbSerial.Location = new Point(27, 8);
            CbSerial.Name = "CbSerial";
            CbSerial.Size = new Size(69, 25);
            CbSerial.TabIndex = 0;
            CbSerial.Click += CbSerial_Click;
            // 
            // pictureBox1
            // 
            pictureBox1.Location = new Point(107, 8);
            pictureBox1.Name = "pictureBox1";
            pictureBox1.Size = new Size(40, 25);
            pictureBox1.SizeMode = PictureBoxSizeMode.CenterImage;
            pictureBox1.TabIndex = 1;
            pictureBox1.TabStop = false;
            // 
            // lbConnState
            // 
            lbConnState.AutoSize = true;
            lbConnState.Font = new Font("Microsoft YaHei UI", 12F, FontStyle.Regular, GraphicsUnit.Point, 134);
            lbConnState.Location = new Point(158, 10);
            lbConnState.Name = "lbConnState";
            lbConnState.Size = new Size(58, 21);
            lbConnState.TabIndex = 2;
            lbConnState.Text = "未连接";
            // 
            // Number
            // 
            Number.AutoSize = true;
            Number.Font = new Font("Microsoft YaHei UI", 12F, FontStyle.Regular, GraphicsUnit.Point, 134);
            Number.Location = new Point(3, 30);
            Number.Name = "Number";
            Number.Size = new Size(19, 21);
            Number.TabIndex = 3;
            Number.Text = "1";
            // 
            // CbSignalType
            // 
            CbSignalType.DropDownStyle = ComboBoxStyle.DropDownList;
            CbSignalType.FormattingEnabled = true;
            CbSignalType.Items.AddRange(new object[] { "EEG", "EMG", "ECG" });
            CbSignalType.Location = new Point(27, 49);
            CbSignalType.Name = "CbSignalType";
            CbSignalType.Size = new Size(69, 25);
            CbSignalType.TabIndex = 5;
            // 
            // CbRate
            // 
            CbRate.DropDownStyle = ComboBoxStyle.DropDownList;
            CbRate.FormattingEnabled = true;
            CbRate.Items.AddRange(new object[] { "500sps", "1000sps", "2000sps" });
            CbRate.Location = new Point(105, 49);
            CbRate.Name = "CbRate";
            CbRate.Size = new Size(73, 25);
            CbRate.TabIndex = 6;
            // 
            // CbRange
            // 
            CbRange.DropDownStyle = ComboBoxStyle.DropDownList;
            CbRange.FormattingEnabled = true;
            CbRange.Items.AddRange(new object[] { "±2.4V", "±1.2V", "±800mV", "±600mV", "±400mV", "±300mV", "±200mV" });
            CbRange.Location = new Point(185, 49);
            CbRange.Name = "CbRange";
            CbRange.Size = new Size(86, 25);
            CbRange.TabIndex = 7;
            // 
            // Uart
            // 
            AutoScaleDimensions = new SizeF(7F, 17F);
            AutoScaleMode = AutoScaleMode.Font;
            Controls.Add(CbRange);
            Controls.Add(CbRate);
            Controls.Add(CbSignalType);
            Controls.Add(Number);
            Controls.Add(lbConnState);
            Controls.Add(pictureBox1);
            Controls.Add(CbSerial);
            Name = "Uart";
            Size = new Size(286, 87);
            Load += Uart_Load;
            ((System.ComponentModel.ISupportInitialize)pictureBox1).EndInit();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion
        private PictureBox pictureBox1;
        private Label lbConnState;
        public Label Number;
        public ComboBox CbSerial;
        private Label label2;
        private ToolTip toolTip1;
        public ComboBox CbSignalType;
        public ComboBox CbRate;
        public ComboBox CbRange;
    }
}
