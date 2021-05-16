using IronPython.Hosting;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Microsoft.Scripting.Hosting;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace ChurnAPI.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class ChurnController : ControllerBase
    {
        private readonly string ModelPath = @"C:/Users/Shalinda/source/repos/shalindasilva1/ML-Project/Controllers/ModelController.py";
        private readonly string PythonPath = @"C:/Users/Shalinda/AppData/Local/Programs/Python/Python39/python.exe";

        private readonly ILogger<ChurnController> _logger;

        public ChurnController(ILogger<ChurnController> logger)
        {
            _logger = logger;
        }

        [HttpGet]
        public string Predict()
        {            
            return PatchParameter(@"C:\Users\Shalinda\source\repos\shalindasilva1\ML-Project\Controllers\test.csv");
        }

        private string PatchParameter(string args)
        {
            string result = string.Empty;
            string errors = string.Empty;
            try
            {
                // create process
                var info = new ProcessStartInfo();
                info.FileName = PythonPath;

                // provide script and arguments
                info.Arguments = $"\"{ModelPath}\" \"{args}\"";

                // process start info settings
                info.UseShellExecute = false;
                info.CreateNoWindow = true;
                info.RedirectStandardOutput = true;
                info.RedirectStandardError = true;

                using (var proc = Process.Start(info))
                {
                    result = proc.StandardOutput.ReadToEnd();
                    errors = proc.StandardError.ReadToEnd();
                }
                return result;
            }
            catch (Exception ex)
            {
                throw new Exception("R Script failed: " + result, ex);
            }
        }
    }
}
