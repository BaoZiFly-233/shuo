# GitCode 仓库管理

## 创建新仓库

### API 方式（推荐脚本化）

使用 GitCode v5 API（注意：不是 `/api/v1`，那个已被 WAF 拦截）：

```powershell
$token = "<personal-access-token>"
$headers = @{
    "Authorization" = "Bearer $token"
    "Accept" = "application/json"
    "Content-Type" = "application/json"
}
$body = @{name="<repo-name>"; private=$false}
Invoke-RestMethod -Uri "https://api.gitcode.com/api/v5/user/repos" `
    -Method Post -Headers $headers -Body ($body | ConvertTo-Json)
```

### 推送代码（SSH）

```powershell
git remote add origin git@gitcode.com:gcw_aU1BuMs3/<repo-name>.git
git push -u origin master
```

### 关键信息

- API 基地址：`https://api.gitcode.com/api/v5`（**不是** `gitcode.com/api/v5`）
- 认证方式：`Authorization: Bearer <token>` 或 `PRIVATE-TOKEN: <token>`
- 用户登录名：`gcw_aU1BuMs3`
- 用户名：`a2heng`
- SSH 可用，已配置密钥
- 状态码 418 = "请求疑似不安全"（CloudWAF 拦截），换用 `api.gitcode.com` 子域名可绕过
