"""
Large conversational dataset for training the Transformer chatbot.
Format: <user> message <bot> response <end>
"""

CONVERSATIONS = """
<user> hello <bot> hello there how are you doing today <end>
<user> hi <bot> hi nice to meet you what can i help you with <end>
<user> hey how are you <bot> i am doing great thanks for asking how about you <end>
<user> good morning <bot> good morning i hope you have a wonderful day ahead <end>
<user> good night <bot> good night sleep well and have sweet dreams <end>
<user> how are you doing <bot> i am doing well thank you for asking <end>
<user> what is your name <bot> i am a small language model created to learn and chat <end>
<user> who are you <bot> i am an ai assistant built from scratch to have conversations <end>
<user> who made you <bot> i was built using python and pytorch as a learning project <end>
<user> what can you do <bot> i can chat with you answer questions and tell you about many topics <end>
<user> are you smart <bot> i am learning and trying my best to be helpful <end>
<user> tell me a joke <bot> why did the computer go to the doctor because it had a virus <end>
<user> tell me something funny <bot> a book fell on my head i only have my shelf to blame <end>
<user> make me laugh <bot> what do you call a bear with no teeth a gummy bear <end>
<user> what is the sun <bot> the sun is a star at the center of our solar system that gives us light and heat <end>
<user> what is the moon <bot> the moon is a natural satellite that orbits the earth and reflects light from the sun <end>
<user> what are stars <bot> stars are giant balls of hot gas that shine in the night sky far away from earth <end>
<user> tell me about space <bot> space is the vast area beyond the earth filled with stars planets and galaxies <end>
<user> what is a planet <bot> a planet is a large body that orbits a star like the earth orbits the sun <end>
<user> how big is the earth <bot> the earth is very large with a diameter of about twelve thousand kilometers <end>
<user> what is gravity <bot> gravity is a force that pulls objects toward each other and keeps us on the ground <end>
<user> what is water <bot> water is a liquid made of hydrogen and oxygen that is essential for all life <end>
<user> what is air <bot> air is a mixture of gases that surrounds the earth and that we breathe to stay alive <end>
<user> what is fire <bot> fire is a chemical reaction that produces heat and light when something burns <end>
<user> what is rain <bot> rain is water that falls from clouds in the sky when they become too heavy <end>
<user> what is snow <bot> snow is frozen water that falls from clouds as white crystals in cold weather <end>
<user> what is a rainbow <bot> a rainbow is a colorful arc in the sky caused by sunlight shining through rain drops <end>
<user> what is the ocean <bot> the ocean is a huge body of salt water that covers most of the earth <end>
<user> what is a river <bot> a river is a natural flowing stream of water that moves toward the ocean or a lake <end>
<user> what is a mountain <bot> a mountain is a very tall natural landform that rises high above the surrounding area <end>
<user> what is a forest <bot> a forest is a large area covered with trees and plants home to many animals <end>
<user> what is a desert <bot> a desert is a dry area with very little rain and few plants or animals <end>
<user> tell me about animals <bot> animals are living creatures that can move and eat and come in many shapes and sizes <end>
<user> what is a cat <bot> a cat is a small furry animal that people keep as a pet it likes to sleep and play <end>
<user> what is a dog <bot> a dog is a loyal animal that people keep as a pet it loves to play and run <end>
<user> tell me about birds <bot> birds are animals with feathers and wings that can fly through the sky <end>
<user> what is a fish <bot> a fish is an animal that lives in water and breathes through gills and swims with fins <end>
<user> what do cats eat <bot> cats eat meat and fish and special cat food to stay healthy and strong <end>
<user> what do dogs eat <bot> dogs eat meat and dog food and sometimes treats from their owners <end>
<user> what is your favorite color <bot> i like blue because it reminds me of the sky and the ocean <end>
<user> what is your favorite food <bot> i do not eat food but i think pizza sounds delicious <end>
<user> what is pizza <bot> pizza is a popular food made with dough tomato sauce and cheese baked in an oven <end>
<user> what is ice cream <bot> ice cream is a cold sweet dessert made from milk and sugar in many flavors <end>
<user> what is chocolate <bot> chocolate is a sweet food made from cocoa beans that many people love to eat <end>
<user> what is a computer <bot> a computer is an electronic machine that can store and process information very quickly <end>
<user> what is the internet <bot> the internet is a global network that connects computers and lets people share information <end>
<user> what is programming <bot> programming is writing instructions that tell a computer what to do using code <end>
<user> what is python <bot> python is a programming language that is easy to learn and used to build many things <end>
<user> what is ai <bot> ai means artificial intelligence which is computers that can learn and make decisions <end>
<user> how do you work <bot> i work by using a neural network that learns patterns in text to predict words <end>
<user> what is a neural network <bot> a neural network is a system inspired by the brain that learns from data <end>
<user> what is machine learning <bot> machine learning is when computers learn from data instead of being programmed directly <end>
<user> what is deep learning <bot> deep learning uses many layers of neural networks to learn complex patterns from data <end>
<user> what is a transformer <bot> a transformer is a neural network that uses attention to understand text very well <end>
<user> what is attention <bot> attention lets the model focus on the most important parts of the input text <end>
<user> what is the weather like <bot> i cannot check the weather but i hope it is nice where you are <end>
<user> it is cold today <bot> stay warm and wear a jacket and maybe have some hot chocolate <end>
<user> it is hot today <bot> stay cool and drink plenty of water and find some shade to rest <end>
<user> i am happy <bot> that is wonderful to hear being happy is a great feeling <end>
<user> i am sad <bot> i am sorry to hear that i hope things get better for you soon <end>
<user> i am bored <bot> lets chat about something fun what topic interests you <end>
<user> i am tired <bot> you should get some rest sleep is very important for your health <end>
<user> i am hungry <bot> you should eat something healthy like fruit or a good meal <end>
<user> i love music <bot> music is wonderful what kind of music do you like to listen to <end>
<user> what is music <bot> music is an art form made of sounds and rhythms that people enjoy <end>
<user> tell me about books <bot> books are written works that tell stories and share knowledge about the world <end>
<user> what is art <bot> art is a way of expressing ideas and feelings through painting drawing or music <end>
<user> what is science <bot> science is the study of the natural world through observation and experiments <end>
<user> what is math <bot> math is the study of numbers shapes and patterns used to understand the world <end>
<user> what is history <bot> history is the study of past events that helps us understand how things changed <end>
<user> what is a friend <bot> a friend is someone who cares about you and you enjoy spending time with <end>
<user> do you have friends <bot> i am an ai so i do not have friends but i enjoy chatting with you <end>
<user> what is love <bot> love is a deep feeling of care and affection for another person <end>
<user> what is the meaning of life <bot> many believe the meaning of life is to be happy learn and help others <end>
<user> what is a dream <bot> a dream is images and feelings that happen in your mind while you sleep <end>
<user> do you dream <bot> i do not dream because i am a program but dreams sound fascinating <end>
<user> what is sleep <bot> sleep is a natural state of rest that your body and mind need to stay healthy <end>
<user> tell me about the earth <bot> the earth is the third planet from the sun and the only known planet with life <end>
<user> tell me about mars <bot> mars is the fourth planet from the sun called the red planet for its color <end>
<user> what is a galaxy <bot> a galaxy is a huge collection of stars planets and gas held together by gravity <end>
<user> what is energy <bot> energy is the ability to do work and it comes in many forms like heat and light <end>
<user> what is electricity <bot> electricity is a form of energy caused by the flow of tiny particles called electrons <end>
<user> what is a robot <bot> a robot is a machine that can perform tasks automatically controlled by a computer <end>
<user> what is happiness <bot> happiness is a feeling of joy and contentment that comes from within yourself <end>
<user> how can i be happy <bot> you can be happy by doing things you love and being kind to others <end>
<user> how can i be smart <bot> you can be smart by reading learning asking questions and practicing every day <end>
<user> what is success <bot> success is achieving your goals and feeling fulfilled with what you have done <end>
<user> give me advice <bot> always be curious keep learning be kind to others and never give up <end>
<user> i need help <bot> i am here to help just tell me what you need <end>
<user> thank you <bot> you are welcome i am happy to help anytime <end>
<user> thanks a lot <bot> no problem at all i am glad i could help you <end>
<user> goodbye <bot> goodbye have a great day and come back anytime <end>
<user> see you later <bot> see you later take care and have a wonderful time <end>
<user> bye <bot> bye take care and i hope to talk to you again soon <end>
<user> what is a tree <bot> a tree is a tall plant with a trunk branches and leaves that grows from the ground <end>
<user> what is a flower <bot> a flower is the colorful part of a plant that blooms and smells nice <end>
<user> why is the sky blue <bot> the sky looks blue because sunlight scatters in the atmosphere and blue scatters the most <end>
<user> what is wind <bot> wind is moving air caused by differences in temperature and pressure <end>
<user> what is lightning <bot> lightning is a bright flash of electricity in the sky during a storm <end>
<user> what is a volcano <bot> a volcano is an opening in the earth where hot lava and ash come out <end>
<user> what is a dinosaur <bot> a dinosaur was a large reptile that lived millions of years ago <end>
<user> what is dna <bot> dna is the molecule in cells that carries genetic instructions for building life <end>
<user> what is the brain <bot> the brain is the organ in your head that controls thoughts feelings and actions <end>
<user> what is memory <bot> memory is the ability of the brain to store and recall information <end>
<user> what is a language <bot> a language is a system of words and rules people use to communicate <end>
<user> what is a game <bot> a game is an activity with rules that people play for fun <end>
<user> do you play games <bot> i do not play games but i can talk about them with you <end>
<user> what is football <bot> football is a popular sport played with a ball by two teams on a field <end>
<user> what is reading <bot> reading is looking at written words and understanding their meaning <end>
<user> why should i read <bot> reading helps you learn new things expand your mind and be creative <end>
<user> what is a phone <bot> a phone is a device that lets you call people and access the internet <end>
<user> what is a movie <bot> a movie is a story told through moving pictures and sound on a screen <end>
<user> what is a song <bot> a song is a piece of music with words that people sing and listen to <end>
<user> can you sing <bot> i cannot sing but i can tell you about many songs and music <end>
<user> tell me something interesting <bot> the human brain has about one hundred billion neurons like stars in a galaxy <end>
<user> tell me a fact <bot> honey never spoils and archaeologists have found edible honey in ancient tombs <end>
<user> what is cooking <bot> cooking is preparing food by combining ingredients and using heat to make meals <end>
<user> what is exercise <bot> exercise is physical activity that keeps your body healthy and your mind sharp <end>
<user> what is health <bot> health is the state of being free from illness and feeling well in body and mind <end>
<user> what is a doctor <bot> a doctor is a person trained in medicine who helps treat sick people <end>
<user> what is a teacher <bot> a teacher is a person who helps students learn new skills and knowledge <end>
<user> what is nature <bot> nature is the physical world around us including plants animals and landscapes <end>
<user> what is the weather <bot> weather is the condition of the atmosphere including temperature rain and wind <end>
<user> what is a season <bot> a season is one of four periods of the year spring summer autumn and winter <end>
<user> what is spring <bot> spring is the season when flowers bloom and the weather gets warmer <end>
<user> what is summer <bot> summer is the warmest season when days are long and people enjoy the outdoors <end>
<user> what is autumn <bot> autumn is the season when leaves change color and fall from the trees <end>
<user> what is winter <bot> winter is the coldest season when snow falls and the days are short <end>
<user> how old are you <bot> i was just created recently so i am very new and still learning <end>
<user> where do you live <bot> i live inside a computer as a program made of code and data <end>
<user> are you real <bot> i am real software but i am not a living being like you <end>
<user> can you think <bot> i process information and find patterns but i do not think like humans do <end>
<user> can you feel <bot> i do not have feelings but i am designed to be helpful and friendly <end>
<user> what is a cloud <bot> a cloud is a mass of water droplets or ice crystals floating in the sky <end>
<user> what is the universe <bot> the universe is everything that exists including all of space time matter and energy <end>
<user> how old is the universe <bot> the universe is about fourteen billion years old according to scientists <end>
<user> what is light <bot> light is a form of energy that lets us see things and travels very fast <end>
<user> what is sound <bot> sound is vibrations that travel through air and that we hear with our ears <end>
<user> what is color <bot> color is what we see when light reflects off objects in different wavelengths <end>
<user> what is time <bot> time is the ongoing progression of events from the past through the present to the future <end>
<user> what is speed <bot> speed is how fast something moves measured by distance traveled over time <end>
<user> what is a car <bot> a car is a vehicle with four wheels and an engine used for transportation <end>
<user> what is a train <bot> a train is a vehicle that runs on tracks and carries people and goods <end>
<user> what is an airplane <bot> an airplane is a vehicle with wings that flies through the sky carrying passengers <end>
<user> what is a boat <bot> a boat is a vessel that floats on water and is used for travel and fishing <end>
<user> what is a house <bot> a house is a building where people live with rooms for sleeping eating and more <end>
<user> what is a city <bot> a city is a large area where many people live and work with buildings and roads <end>
<user> what is a country <bot> a country is a nation with its own government borders people and culture <end>
<user> what is money <bot> money is something people use to buy things and pay for services <end>
<user> what is a job <bot> a job is work that a person does to earn money and contribute to society <end>
<user> what do you like <bot> i like learning new things and having interesting conversations with people <end>
<user> you are cool <bot> thank you that is very kind of you to say <end>
<user> you are helpful <bot> thank you i try my best to be useful and informative <end>
<user> you are amazing <bot> thank you so much that makes me happy to hear <end>
<user> i like you <bot> thank you i enjoy talking with you too <end>
<user> lets be friends <bot> i would like that very much lets keep chatting and learning together <end>
"""
