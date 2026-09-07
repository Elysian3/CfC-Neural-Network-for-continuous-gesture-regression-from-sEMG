using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RunE_Module
{
    public class RtJson
    {
        /// <summary>
        /// 读取JSON文件
        /// </summary>
        public JObject? Readjson(string path)
        {
            if (System.IO.File.Exists(path))
            {
                try
                {
                    using (System.IO.StreamReader streamReader = System.IO.File.OpenText(path))
                    {
                        using (JsonTextReader reader = new JsonTextReader(streamReader))
                        {
                            JObject jObject = (JObject)JToken.ReadFrom(reader);
                            return jObject;
                        }
                    }
                }
                catch
                {
                    return null;
                }
            }
            return null;   
        }

        /// <summary>
        /// 写JSON文件
        /// </summary>
        /// <param name="jsonfile"></param>
        /// <param name="jObject"></param>
        public void Writejson(string path, JObject jObject)
        {
            using (System.IO.StreamWriter streamWriter = new System.IO.StreamWriter(path))
            {
                streamWriter.Write(jObject.ToString());
            }
        }
    }
}
