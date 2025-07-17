from src.agents.critic_agent import create_critic_agent, create_critic_from_env
from src.agents.profiler_agent import create_profiler_agent, create_profiler_from_env
from src.agents.evaluator_agents import create_all_evaluator_agents, create_evaluators_from_env

# Тест 1: Critic Agent
critic = create_critic_agent()
print("✅ Critic agent модель:", critic.config.llm_config.model)

# Тест 2: Profiler Agent  
profiler = create_profiler_agent()
print("✅ Profiler agent модель:", profiler.config.llm_config.model)

# Тест 3: Evaluator Agents
evaluators = create_all_evaluator_agents()
print("✅ Evaluator agents модель:", next(iter(evaluators.values())).config.llm_config.model)

# Тест 4: From env функции
critic_env = create_critic_from_env()
print("✅ Critic from env модель:", critic_env.config.llm_config.model)

print("🎯 Все специализированные агенты используют центральную конфигурацию!")