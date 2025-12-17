# 移除邮箱和名称要求 - 修改总结

## 问题
1. 数据库 auth 表使用 email 列，代码期望 username 列
2. user 表的 email 字段不允许 NULL
3. 前端注册时要求填写名称

## 已完成的修改

### 1. 数据库修复 (已手动执行)
- ✅ 修复 auth 表：将 email 列改为 username 列
- ✅ 修复 user 表：将 username 设为必需和唯一，email 设为可选

### 2. 后端代码修改

**backend/open_webui/models/users.py:**
- ✅ User 模型: email 改为 nullable=True
- ✅ UserUpdateForm: email 改为可选
- ✅ UserInfoResponse: email 改为可选
- ✅ UserResponse: email 改为可选
- ✅ UserProfileImageResponse: email 改为可选

**backend/open_webui/routers/auths.py:**
- ✅ signup: 不再需要 email

**backend/open_webui/routers/users.py:**
- ✅ 修复更新用户时的 email 比较逻辑，处理 None 情况
- ✅ 移除不存在的 Auths.update_email_by_id() 调用
- ✅ 使 email 更新变为可选

**backend/open_webui/routers/scim.py:**
- ✅ 当 email 为 None 时，使用 username 作为后备值

**backend/open_webui/routers/files.py:**
- ✅ 文件上传时，如果 email 为 None，使用 username 作为后备值

### 3. 前端修改

**src/routes/auth/+page.svelte:**
- ✅ signUpHandler: 如果不填写名称，自动使用用户名作为名称
- ✅ 名称输入框: 移除 required 属性，变为可选
- ✅ 更新标签和占位符文本，提示用户可以留空

## 现在的使用方式

### 注册
用户只需要填写：
- **用户名** (必填)
- **密码** (必填)
- **名称** (可选，留空则使用用户名)

### 登录
用户只需要：
- **用户名**
- **密码**

## 测试
请重启服务器后测试：
```bash
# 停止当前服务器 (Ctrl+C)
cd backend
dev.bat
```

然后访问注册页面，应该可以：
1. 只填写用户名和密码注册
2. 名称字段是可选的
3. 不需要填写邮箱
