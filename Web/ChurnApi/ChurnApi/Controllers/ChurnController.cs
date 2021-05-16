using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using System;
using System.Diagnostics;
using System.IO;

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

        [HttpPost]
        public string Predict(IFormFile inputFile)
        {
            var directoryInfo = TryToCreateNewSessionFolder();
            var inputFilePath = SaveFile(inputFile, directoryInfo.Item1.FullName);
            return PatchParameter(inputFilePath, directoryInfo.Item2);
        }

        private string SaveFile(IFormFile file, string sessionFolderPath)
        {
            if (!Directory.Exists(Path.Combine(sessionFolderPath, "input")))
            {
                Directory.CreateDirectory(Path.Combine(sessionFolderPath, "input"));
            }
            string filePath = Path.Combine(sessionFolderPath, "input", file.FileName);
            using (var x = System.IO.File.Create(filePath))
            {
                file.CopyTo(x);
            }
            return filePath;
        }

        private (DirectoryInfo, string) TryToCreateNewSessionFolder()
        {
            Random random = new Random();
            var sessionGUID = random.Next();
            string path = Path.Combine(Directory.GetCurrentDirectory(),sessionGUID.ToString());
            DirectoryInfo di;
            while (true)
            {
                if (!Directory.Exists(path))
                {
                    di = Directory.CreateDirectory(path);
                    break;
                }
                sessionGUID = random.Next();
                path = Path.Combine(Directory.GetCurrentDirectory(), sessionGUID.ToString());
            }
            return (di, sessionGUID.ToString());
        }

        private string PatchParameter(string fileName, string sessionId)
        {
            string result = string.Empty;
            string errors = string.Empty;
            try
            {
                // create process
                var info = new ProcessStartInfo();
                info.FileName = PythonPath;

                // provide script and arguments
                info.Arguments = $"\"{ModelPath}\" \"{fileName}\" \"{sessionId}\"";

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
