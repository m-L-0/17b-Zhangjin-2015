//
// var filechange=function(event){
//     var files = event.target.files, file;
//     if (files && files.length > 0) {
//         // 获取目前上传的文件
//         file = files[0];// 文件大小校验的动作
//         if(file.size > 1024 * 1024 * 2) {
//             alert('图片大小不能超过 2MB!');
//             return false;
//         }
//         // 获取 window 的 URL 工具
//         var URL = window.URL || window.webkitURL;
//         // 通过 file 生成目标 url
//         var imgURL = URL.createObjectURL(file);
//         //用attr将img的src属性改成获得的url
//         $("#img-change").attr("src",imgURL);
//         // 使用下面这句可以在内存中释放对此 url 的伺服，跑了之后那个 URL 就无效了
//         // URL.revokeObjectURL(imgURL);
//     }
// };
//
// $("#btn").click(function () {
//         $.ajaxFileUpload({
//             url: '/imgUpload',
//             fileElementId:'file',
//             dataType:'txt',
//             secureuri : false,
//             success: function (data){
//                 if(data=="yes"){
//                     $("#img-alert").css("display","block");
//                 }
//             },
//             error:function(data,status,e){
//                 alert(e);
//             }
//         });
//     });

